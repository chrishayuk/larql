//! Expert-bank binding and production-backend arms not reached by the parity gates.
//!
//! Two groups. **Expert-bank binding** (`experts.rs`): the packed bank's
//! refusal arms — a projection without scales, a stored dtype or byte
//! count that disagrees with the declared geometry, an unaligned `k`, a
//! per-expert format that is not packed — plus the conversion arms a
//! non-f32 backend takes (MXFP4 → f16/NVFP4, BF16 → f16/f32/MXFP4/NVFP4)
//! and the residency listing. **Production backend** (`production.rs`):
//! the arms the routed and dense fixtures never select — weighted QK
//! norm, `Windowed` span, LayerNorm, the plain projection, embedding
//! scale, the Gemma 4 router refusal, the bias-free router, the
//! ungated FFN, and every kernel refusal.
//!
//! The routed fixture is the GPT-OSS miniature of `routed.rs` with one
//! twist: every source value is bf16-representable, so an unquantised
//! BF16 copy of the bank (written under unclassified names, into the bank
//! object) quantises at load to **exactly** the bytes the checkpoint's own
//! MXFP4 packing holds — the BF16 → MXFP4 arm is checked byte for byte,
//! not against a tolerance.

mod fixture;

use larql_models::config::{
    Activation, ExpertFormat, MoeRouterKind, NormType, ParameterFreeQkNorm, PositionPolicy,
    QkNormScope,
};
use larql_models::ExpertGatePolicy;

use self::fixture::*;
use super::{dense_f32_model, lcg_values, norm_values};
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionStepCall, FfnCall, NormCall, PlanBackend, ProjectCall, QkNormCall,
    RoutedFfnCall, WeightFormat, WeightSlice,
};
use crate::format::vindex3::opplan::exec::experts::FfnOperands;
use crate::format::vindex3::opplan::exec::production::{
    aggregate_heads, condition_qk_in_place, qk_norm_in_place, select_experts, ProductionBackend,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{FfnOp, LayerFfn};

// ═══════════════════════════ experts.rs ═══════════════════════════

/// The residency listing of a routed layer is every expert's gate_up and
/// down matrix — one slice per (expert, projection), in the declared
/// format — so `prepare` can wire the whole bank.
#[test]
fn routed_operands_list_every_expert_matrix_for_residency() {
    let fx = routed_fixture();
    let operands = load(&fx.op, &fx.store, WeightFormat::F32);
    let kinds = slice_kinds(&operands);
    assert_eq!(kinds.len(), FUSED_BRANCHES * EXPERTS);
    assert!(kinds.iter().all(|k| *k == "f32"), "{kinds:?}");
}

/// A routed op without a `gate_up_layout` refuses at apply time — the
/// layout is what makes the fused rows readable — naming the missing fact.
#[test]
fn a_routed_op_without_a_gate_up_layout_refuses_at_apply() {
    let fx = routed_fixture();
    let operands = load(&fx.op, &fx.store, WeightFormat::F32);
    let mut op = fx.op.clone();
    op.gate_up_layout = None;
    let x = norm_values(HIDDEN, INPUT_SEED);
    let err = operands
        .apply(&routed(&op), &ProductionBackend::new(), &x, HIDDEN)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no gate_up layout"), "{err}");
}

/// Operands loaded for a routed op refuse to serve a dense op: the two
/// kinds share nothing, and a mismatch is an interpreter bug, not a
/// fallback.
#[test]
fn operands_loaded_for_one_ffn_kind_refuse_the_other_at_apply() {
    let fx = routed_fixture();
    let operands = load(&fx.op, &fx.store, WeightFormat::F32);
    let dense = LayerFfn::Dense(Box::new(FfnOp {
        intermediate_size: INTER,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        gate: None,
        up: fx.op.router.clone(),
        down: fx.op.router.clone(),
    }));
    let x = norm_values(HIDDEN, INPUT_SEED);
    let err = operands
        .apply(&dense, &ProductionBackend::new(), &x, HIDDEN)
        .unwrap_err()
        .to_string();
    assert!(err.contains("different op kind"), "{err}");
}

/// An MXFP4 projection whose op carries no scales operand refuses,
/// naming the projection — the blocks alone are not a matrix.
#[test]
fn an_mxfp4_projection_without_scales_refuses() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.gate_up.scales = None;
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("no scales operand"), "{err}");
    assert!(err.contains(GATE_UP_BLOCKS), "{err}");
}

/// An MXFP4 projection whose `k` is not a multiple of the 32-element
/// group refuses before touching the bytes: `k` follows from the router's
/// declared width, and a stray width must not be silently grouped.
#[test]
fn an_mxfp4_projection_with_an_unaligned_k_refuses() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.router.shape[1] = HIDDEN + 1;
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("not a multiple of the MXFP4 group"), "{err}");
}

/// A scales stream of the wrong length for the declared expert geometry
/// refuses, naming the scales tensor: here gate_up's scales are pointed
/// at down's (half the size).
#[test]
fn a_scales_stream_of_the_wrong_length_refuses() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.gate_up.scales = op.down.scales.clone();
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("stored bytes, expected"), "{err}");
    assert!(
        err.contains(DOWN_BLOCKS.replace(BLOCKS_SUFFIX, SCALES_SUFFIX).as_str()),
        "{err}"
    );
}

/// A block stream whose byte count disagrees with the declared expert
/// geometry refuses (the op's intermediate width was edited, the bytes
/// were not).
#[test]
fn a_block_stream_disagreeing_with_the_declared_geometry_refuses() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.expert_intermediate_size = INTER + MXFP4_GROUP_ELEMS;
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("stored bytes, expected"), "{err}");
    assert!(err.contains(GATE_UP_BLOCKS), "{err}");
}

/// A stream stored in a dtype other than the one its format demands
/// refuses, naming both dtypes: an MXFP4 scales operand pointed at the
/// F32 bias, and a BF16 projection pointed at the U8 blocks.
#[test]
fn a_stream_stored_in_the_wrong_dtype_refuses() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.gate_up.scales = op.gate_up.bias.clone();
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("expected stored dtype U8, found F32"), "{err}");

    let mut op = fx.op.clone();
    op.expert_format = ExpertFormat::PackedBF16;
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(
        err.contains("expected stored dtype BF16, found U8"),
        "{err}"
    );
}

/// `PerExpert` is not a packed projection; closure never plans one, and
/// the loader refuses rather than guessing a layout.
#[test]
fn a_per_expert_format_is_refused_as_not_packed() {
    let fx = routed_fixture();
    let mut op = fx.op.clone();
    op.expert_format = ExpertFormat::PerExpert;
    let err = load_err(&op, &fx.store, WeightFormat::F32);
    assert!(err.contains("not a packed projection"), "{err}");
}

/// A backend that declares f16 for the FFN class receives the MXFP4 bank
/// converted through f32 to f16 (e2m1 × 2ⁿ is exact in f16 at these
/// magnitudes), and computes the same routed output as the f32 backend;
/// an NVFP4 declaration receives NVFP4 streams.
#[test]
fn an_mxfp4_bank_converts_through_f32_for_f16_and_nvfp4_backends() {
    let fx = routed_fixture();
    let x = norm_values(HIDDEN, INPUT_SEED);
    let ffn = routed(&fx.op);
    let f32_out = load(&fx.op, &fx.store, WeightFormat::F32)
        .apply(&ffn, &ProductionBackend::new(), &x, HIDDEN)
        .unwrap();

    let f16 = load(&fx.op, &fx.store, WeightFormat::F16);
    assert!(slice_kinds(&f16).iter().all(|k| *k == "f16"));
    let f16_out = f16
        .apply(&ffn, &loop_device(WeightFormat::F16), &x, HIDDEN)
        .unwrap();
    let delta = max_abs_diff(&f32_out, &f16_out);
    assert!(delta < NOISE_CEILING, "f16 realisation drifted {delta:e}");

    let nvfp4 = load(&fx.op, &fx.store, WeightFormat::Nvfp4);
    let kinds = slice_kinds(&nvfp4);
    assert_eq!(kinds.len(), FUSED_BRANCHES * EXPERTS);
    assert!(kinds.iter().all(|k| *k == "nvfp4"), "{kinds:?}");
}

/// A BF16 bank binds in every declared format: f16 exactly (bf16 ⊂ f16),
/// f32 by widening, and MXFP4/NVFP4 by quantising — where the MXFP4
/// bytes equal the checkpoint's own packing of the same values, byte for
/// byte, because the sources are bf16-representable.
#[test]
fn a_bf16_bank_binds_in_every_declared_format() {
    let fx = routed_fixture();
    let (_dir, _container, store) = bf16_carrier_store();
    let op = bf16_op(&fx.op);
    let ffn = routed(&op);
    let x = norm_values(HIDDEN, INPUT_SEED);

    let f32_bank = load(&op, &store, WeightFormat::F32);
    assert!(slice_kinds(&f32_bank).iter().all(|k| *k == "f32"));
    let f32_out = f32_bank
        .apply(&ffn, &ProductionBackend::new(), &x, HIDDEN)
        .unwrap();

    let f16_bank = load(&op, &store, WeightFormat::F16);
    assert!(slice_kinds(&f16_bank).iter().all(|k| *k == "f16"));
    let f16_out = f16_bank
        .apply(&ffn, &loop_device(WeightFormat::F16), &x, HIDDEN)
        .unwrap();
    let delta = max_abs_diff(&f32_out, &f16_out);
    assert!(delta < NOISE_CEILING, "f16 realisation drifted {delta:e}");

    let from_bf16 = mxfp4_bytes(&load(&op, &store, WeightFormat::Mxfp4));
    let native = mxfp4_bytes(&load(&fx.op, &store, WeightFormat::Mxfp4));
    assert_eq!(from_bf16.len(), FUSED_BRANCHES * EXPERTS);
    assert_eq!(
        from_bf16, native,
        "BF16 bank quantised at load must equal the checkpoint's MXFP4 packing"
    );

    let nvfp4 = slice_kinds(&load(&op, &store, WeightFormat::Nvfp4));
    assert!(nvfp4.iter().all(|k| *k == "nvfp4"), "{nvfp4:?}");
}

/// A BF16 bank whose byte count disagrees with the declared geometry
/// refuses, naming the projection.
#[test]
fn a_bf16_bank_disagreeing_with_the_declared_geometry_refuses() {
    let fx = routed_fixture();
    let (_dir, _container, store) = bf16_carrier_store();
    let mut op = bf16_op(&fx.op);
    op.expert_intermediate_size = INTER + 1;
    let err = load_err(&op, &store, WeightFormat::F32);
    assert!(err.contains("stored bytes, expected"), "{err}");
    assert!(err.contains(GATE_UP_BF16), "{err}");
}

/// A dense op without a gate binds two operands, and its ungated SiLU
/// runs through the production `silu` — agreeing with the reference's
/// scalar transcription.
#[test]
fn a_gate_less_dense_op_binds_two_operands_and_runs_ungated() {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let container = encoded(dir.path(), "dense");
    let (plan, store) = closed_plan(container.path());
    let mut op = plan.layers[0].ffn.dense().expect("dense layer").clone();
    op.gate = None;
    let ffn = LayerFfn::Dense(Box::new(op));
    let operands = FfnOperands::load(&ffn, &store, WeightFormat::F32).unwrap();
    assert_eq!(operands.weight_slices().len(), 2, "up and down only");
    let x = norm_values(super::HIDDEN, INPUT_SEED);
    let production = operands
        .apply(&ffn, &ProductionBackend::new(), &x, super::HIDDEN)
        .unwrap();
    let reference = operands
        .apply(&ffn, &ReferenceBackend::new(), &x, super::HIDDEN)
        .unwrap();
    let delta = max_abs_diff(&production, &reference);
    assert!(delta < NOISE_CEILING, "ungated FFN drifted {delta:e}");
}

// ═══════════════════════════ production.rs ═══════════════════════════

/// A `[2, 2]` identity-shaped weight for the tiny direct calls.
const TINY: usize = 2;
const TINY_WEIGHT: [f32; TINY * TINY] = [1.0, 0.0, 0.0, 2.0];
const TINY_X: [f32; TINY] = [3.0, 4.0];

/// The backend names itself for parity reports, and the embedding scale
/// is applied when the plan carries one — a scaled row is the unscaled
/// row times the scale, not a re-lookup.
#[test]
fn the_production_backend_names_itself_and_scales_embeddings() {
    const SCALE: f32 = 0.5;
    let backend = ProductionBackend::new();
    assert_eq!(backend.name(), "production-larql-compute");
    let table = lcg_values(TINY * TINY, INPUT_SEED);
    let plain = backend.embed(&table, TINY, 1, None);
    let scaled = backend.embed(&table, TINY, 1, Some(SCALE));
    assert_eq!(plain.as_slice(), &table[TINY..]);
    let expected: Vec<f32> = plain.iter().map(|v| v * SCALE).collect();
    assert_eq!(scaled, expected);
}

/// LayerNorm binds the production layer-norm kernel (bias absent, not
/// zeroed) and `project` binds `matmul_vec`: `[1, 3]` normalises to
/// `[-1, 1]` up to eps, and the diagonal weight doubles the second lane.
#[test]
fn layer_norm_and_projection_bind_the_production_kernels() {
    let backend = ProductionBackend::new();
    let normed = backend.norm(NormCall {
        kind: NormType::LayerNorm,
        x: &[1.0, 3.0],
        weight: &[1.0, 1.0],
        weight_offset: 0.0,
        eps: 0.0,
    });
    assert!(
        max_abs_diff(&normed, &[-1.0, 1.0]) < NOISE_CEILING,
        "{normed:?}"
    );
    let projected = backend
        .project(ProjectCall {
            weight: WeightSlice::F32(&TINY_WEIGHT),
            out_dim: TINY,
            in_dim: TINY,
            x: &TINY_X,
        })
        .unwrap();
    assert_eq!(projected, vec![3.0, 8.0]);
}

/// A hand-built attention call over `inputs`, with no optional judgment.
fn plain_attention<'a>(inputs: &'a [Vec<f32>], w: &'a [f32]) -> AttentionCall<'a> {
    AttentionCall {
        inputs,
        hidden: HEAD_DIM,
        num_q_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM,
        w_q: WeightSlice::F32(w),
        w_k: WeightSlice::F32(w),
        w_v: WeightSlice::F32(w),
        w_o: WeightSlice::F32(w),
        qk_norm: None,
        parameter_free_qk_norm: ParameterFreeQkNorm {
            q: false,
            k: false,
            v: false,
        },
        qk_norm_eps: EPS,
        query_scale: None,
        score_scale: 1.0 / (HEAD_DIM as f64).sqrt(),
        logit_softcapping: None,
        position: PositionPolicy::None,
        span: AttentionSpan::Full,
        window: None,
        gate: None,
        bias: None,
        sinks: None,
    }
}

/// A weighted QK norm with unit weight (or zero weight and unit offset —
/// the Gemma spelling) is the parameter-free norm: the weighted kernel is
/// bound and the offset is applied, through both the raw helper and the
/// call-level conditioning.
#[test]
fn a_unit_weight_qk_norm_equals_the_parameter_free_form() {
    let raw = lcg_values(Q_HEADS * HEAD_DIM, INPUT_SEED);
    let ones = vec![1.0f32; HEAD_DIM];
    let zeros = vec![0.0f32; HEAD_DIM];

    let mut parameter_free = raw.clone();
    qk_norm_in_place(
        &mut parameter_free,
        None,
        true,
        Q_HEADS,
        HEAD_DIM,
        QkNormScope::PerHead,
        EPS,
    );
    assert!(
        max_abs_diff(&parameter_free, &raw) > NOISE_CEILING,
        "norm must move the vector"
    );
    let mut weighted = raw.clone();
    qk_norm_in_place(
        &mut weighted,
        Some((&ones, 0.0)),
        false,
        Q_HEADS,
        HEAD_DIM,
        QkNormScope::PerHead,
        EPS,
    );
    assert!(max_abs_diff(&weighted, &parameter_free) < NOISE_CEILING);
    let mut offset = raw.clone();
    qk_norm_in_place(
        &mut offset,
        Some((&zeros, 1.0)),
        false,
        Q_HEADS,
        HEAD_DIM,
        QkNormScope::PerHead,
        EPS,
    );
    assert!(max_abs_diff(&offset, &parameter_free) < NOISE_CEILING);

    // Through the call: Q under a unit-weight norm, K under a zero-weight /
    // unit-offset norm, one head each — both equal the parameter-free form.
    let inputs = vec![raw[..HEAD_DIM].to_vec()];
    let identity = vec![0.0f32; HEAD_DIM * HEAD_DIM];
    let mut call = plain_attention(&inputs, &identity);
    call.qk_norm = Some(QkNormCall {
        scope: QkNormScope::PerHead,
        weight_offset: 0.0,
        q_weight: &ones,
        k_weight: &ones,
    });
    let mut q = raw[..HEAD_DIM].to_vec();
    let mut k = raw[HEAD_DIM..].to_vec();
    condition_qk_in_place(&call, 0, &mut q, &mut k).unwrap();
    assert!(max_abs_diff(&q, &parameter_free[..HEAD_DIM]) < NOISE_CEILING);
    assert!(max_abs_diff(&k, &parameter_free[HEAD_DIM..]) < NOISE_CEILING);
}

/// A `Windowed` span carries no sequence bound: with a window count set
/// it still aggregates over the whole prefix, exactly as `Full` does.
#[test]
fn a_windowed_span_aggregates_the_whole_prefix() {
    const POSITIONS: usize = 3;
    let rows: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|p| lcg_values(HEAD_DIM, INPUT_SEED + p as u64))
        .collect();
    let query = &rows[POSITIONS - 1];
    let identity = vec![0.0f32; HEAD_DIM * HEAD_DIM];
    let mut full = plain_attention(&rows, &identity);
    full.span = AttentionSpan::Full;
    let mut windowed = plain_attention(&rows, &identity);
    windowed.span = AttentionSpan::Windowed;
    windowed.window = Some(1);
    let mut sliding = plain_attention(&rows, &identity);
    sliding.span = AttentionSpan::Sliding;
    sliding.window = Some(1);
    let key_of = |p: usize| rows[p].as_slice();
    let full_out = aggregate_heads(&full, POSITIONS - 1, query, key_of, key_of);
    let windowed_out = aggregate_heads(&windowed, POSITIONS - 1, query, key_of, key_of);
    let sliding_out = aggregate_heads(&sliding, POSITIONS - 1, query, key_of, key_of);
    assert_eq!(windowed_out, full_out);
    // Control: the same window count under `Sliding` does bound the span.
    assert!(max_abs_diff(&sliding_out, &full_out) > NOISE_CEILING);
}

/// A routed call over `logits.len()` experts with the given router bias.
fn routed_call<'a>(x: &'a [f32], router: &'a [f32], bias: Option<&'a [f32]>) -> RoutedFfnCall<'a> {
    RoutedFfnCall {
        x,
        hidden: x.len(),
        intermediate: INTER,
        experts: router.len() / x.len(),
        top_k: TOP_K,
        router_kind: MoeRouterKind::TopKThenSoftmax,
        routing_policy: larql_models::config::ExpertRoutingPolicy::NormalisedOverSelected,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        gate_up_layout: larql_models::config::GateUpLayout::Interleaved,
        router,
        router_bias: bias,
        gate_up: &[],
        gate_up_bias: None,
        down: &[],
        down_bias: None,
    }
}

/// The Gemma 4 hybrid router is refused (its per-expert scale is not
/// implemented here), and a call without a router bias routes on the
/// raw logits — identical to a zero bias.
#[test]
fn gemma4_routing_is_refused_and_a_missing_router_bias_adds_nothing() {
    let x = norm_values(HIDDEN, INPUT_SEED);
    let router = lcg_values(EXPERTS * HIDDEN, INPUT_SEED + 1);
    let logits: Vec<f32> = lcg_values(EXPERTS, INPUT_SEED + 2);
    let zeros = vec![0.0f32; EXPERTS];

    let mut hybrid = routed_call(&x, &router, None);
    hybrid.router_kind = MoeRouterKind::Gemma4Hybrid;
    let err = select_experts(&hybrid, &mut logits.clone())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Gemma4Hybrid"), "{err}");

    let unbiased = select_experts(&routed_call(&x, &router, None), &mut logits.clone()).unwrap();
    let zero_biased =
        select_experts(&routed_call(&x, &router, Some(&zeros)), &mut logits.clone()).unwrap();
    assert_eq!(unbiased.len(), TOP_K);
    assert_eq!(unbiased, zero_biased);
}

/// A decode step whose output projection arrives in a representation the
/// backend did not declare fails closed after projection — the error names
/// the representation mismatch rather than converting.
#[test]
fn a_decode_step_fails_closed_on_a_non_f32_output_projection() {
    let inputs = vec![lcg_values(HEAD_DIM, INPUT_SEED)];
    let identity = vec![0.0f32; HEAD_DIM * HEAD_DIM];
    let mut op = plain_attention(&inputs, &identity);
    op.w_o = WeightSlice::F16(&[]);
    let err = ProductionBackend::new()
        .attention_step(AttentionStepCall {
            op,
            position: 0,
            keys: &[],
            values: &[],
        })
        .err()
        .expect("a non-f32 output projection must refuse")
        .to_string();
    assert!(err.contains("declared f32 weights"), "{err}");
}

/// A dense FFN call over the tiny diagonal weights.
fn tiny_ffn<'a>(gate: bool, activation: Activation, policy: ExpertGatePolicy) -> FfnCall<'a> {
    FfnCall {
        x: &TINY_X,
        hidden: TINY,
        intermediate: TINY,
        gate: gate.then_some(WeightSlice::F32(&TINY_WEIGHT)),
        up: WeightSlice::F32(&TINY_WEIGHT),
        down: WeightSlice::F32(&TINY_WEIGHT),
        activation,
        gate_policy: policy,
    }
}

/// The production FFN refuses what `larql-compute` has no kernel for —
/// a non-SiLU activation, gated or not, and the clamped-GLU gate policy —
/// naming the shape and the activation or policy, and runs the ungated
/// SiLU it does have.
#[test]
fn the_production_ffn_refuses_unkernelled_variants_and_runs_ungated_silu() {
    let backend = ProductionBackend::new();
    let err = backend
        .ffn(tiny_ffn(true, Activation::Gelu, ExpertGatePolicy::Gated))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("gated-FFN kernel for activation Gelu"),
        "{err}"
    );
    let err = backend
        .ffn(tiny_ffn(false, Activation::Relu, ExpertGatePolicy::Gated))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("ungated-FFN kernel for activation Relu"),
        "{err}"
    );
    let clamped = ExpertGatePolicy::ClampedGlu {
        limit: SWIGLU_LIMIT,
        alpha: 1.0,
    };
    let err = backend
        .ffn(tiny_ffn(true, Activation::Silu, clamped))
        .unwrap_err()
        .to_string();
    assert!(err.contains("ClampedGlu"), "{err}");

    let ungated = backend
        .ffn(tiny_ffn(false, Activation::Silu, ExpertGatePolicy::Gated))
        .unwrap();
    let reference = ReferenceBackend::new()
        .ffn(tiny_ffn(false, Activation::Silu, ExpertGatePolicy::Gated))
        .unwrap();
    assert!(
        max_abs_diff(&ungated, &reference) < NOISE_CEILING,
        "{ungated:?} vs {reference:?}"
    );
}
