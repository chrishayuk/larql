//! The Gemma 4 arms of the VINDEX3 Metal lowering, isolated from the CLI
//! and judged against CPU references transcribed from the HF /
//! interpreter semantics — never from the encode order.
//!
//! | proof | what it establishes |
//! |---|---|
//! | weighted QK norm | `AttnWeights.qk_norm` applies Gemma's `q_norm`/`k_norm` (per-head RMS × weight) after the projections and before rope, at parity; dropping it moves the output |
//! | V norm | `AttnShape.parameter_free_v` applies a weightless per-head RMS to the raw value projection, at parity; dropping it moves the output |
//! | K≡V binding | with the SAME matrix bound as `k` and `v`, V is the RAW K projection then v-normed — not the weighted-normed K, not the rotated K |
//! | tanh-GELU FFN | `FfnActivation::GeluTanh` selects the tanh-GELU gate at parity with `larql_compute::ffn::gelu_tanh`; the SiLU arm is distinguishable |
//! | hybrid stack | a two-layer hybrid (dense + routed MXFP4 experts) stack with a DIFFERENT head width and rope table per layer matches the CPU reference at every checkpoint; the per-expert scale and the layer scale are live |
//!
//! Every parity claim carries a control that must dwarf the parity
//! residual — agreement alone cannot show the lowering executed the op.

#![cfg(target_os = "macos")]

#[path = "lowering_gemma4_support/mod.rs"]
mod support;

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition, QkNormWeights,
};
use larql_compute_metal::lowering::ffn::{FfnActivation, FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::stack::{
    Checkpoint, HybridFfnLowering, HybridScratch, LayerFfnLowering, LayerLowering,
    RoutedFfnLowering, StackScratch,
};
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_compute_metal::MetalBackend;
use support::hybrid_stack::{
    build_layer, cpu_stack, upload_layer, LayerDevice, StackLayer, Tweak, ATTN_NORM,
    CENTRED_OFFSET, DENSE_INTER, INV_FREQ, K_CACHE, K_NORM, LAYERS, PE_SCALE, POST_DENSE, POST_EPS,
    POST_EXPERTS, POST_FFN, PRE_EXPERTS, PRE_FFN, Q_NORM, ROUTER_COND, STACK_PARITY, S_Q_ROWS,
    V_CACHE,
};
use support::{
    cpu_attention, det, device, f16_matrix, gated_ffn_branch, hybrid_route, near_one, rel_rms,
    run_once, AttnGeometry, AttnOperands, DenseFfn, VSource, EPS, HIDDEN, HIDDEN_AMPLITUDE,
    NORM_WEIGHT_AMPLITUDE, RAW_OFFSET, WEIGHT_AMPLITUDE,
};

// ── shared amplitudes / tolerances ──────────────────────────────────

/// f16 weights, f32 reassociation only.
const FRAGMENT_PARITY: f64 = 1e-4;
/// A dropped op must move the output by at least this, relative, and by
/// `CONTROL_OVER_PARITY`× the parity residual.
const CONTROL_MIN: f64 = 1e-2;
const CONTROL_OVER_PARITY: f64 = 10.0;

fn assert_parity(what: &str, want: &[f32], got: &[f32], bar: f64) -> f64 {
    assert!(got.iter().all(|v| v.is_finite()), "{what}: non-finite");
    let e = rel_rms(want, got);
    eprintln!("{what}: parity rel_rms {e:.3e}");
    assert!(e < bar, "{what} off its reference: {e:.3e}");
    e
}

fn assert_control(what: &str, parity: f64, moved: f64) {
    eprintln!("{what}: control moved rel_rms {moved:.3e} (parity {parity:.3e})");
    assert!(
        moved > CONTROL_MIN && moved > parity * CONTROL_OVER_PARITY,
        "control BLIND: `{what}` moved the output by only {moved:.3e} (parity {parity:.3e})"
    );
}

// ── 1-3. attention arms ─────────────────────────────────────────────

const A_NUM_Q: usize = 4;
const A_NUM_KV: usize = 2;
const A_HEAD_DIM: usize = 16;
const A_Q_ROWS: usize = A_NUM_Q * A_HEAD_DIM;
const A_KV_ROWS: usize = A_NUM_KV * A_HEAD_DIM;
const A_T: usize = 5;
const A_POS: usize = A_T - 1;
const A_THETA: f64 = 10_000.0;

struct AttnFixture {
    h: Vec<f32>,
    norm_w: Vec<f32>,
    q: (Vec<f32>, Vec<u8>),
    k: (Vec<f32>, Vec<u8>),
    v: (Vec<f32>, Vec<u8>),
    o: (Vec<f32>, Vec<u8>),
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    inv_freq: Vec<f32>,
}

fn attn_fixture() -> AttnFixture {
    let mk16 = |n: usize, k: usize, seed: u32| f16_matrix(&det(n * k, seed, WEIGHT_AMPLITUDE));
    AttnFixture {
        h: det(HIDDEN, 1, HIDDEN_AMPLITUDE),
        norm_w: near_one(HIDDEN, 2, NORM_WEIGHT_AMPLITUDE),
        q: mk16(A_Q_ROWS, HIDDEN, 3),
        k: mk16(A_KV_ROWS, HIDDEN, 4),
        v: mk16(A_KV_ROWS, HIDDEN, 5),
        o: mk16(HIDDEN, A_Q_ROWS, 6),
        q_norm: near_one(A_HEAD_DIM, 7, NORM_WEIGHT_AMPLITUDE),
        k_norm: near_one(A_HEAD_DIM, 8, NORM_WEIGHT_AMPLITUDE),
        k_cache: det(A_T * A_KV_ROWS, 9, HIDDEN_AMPLITUDE),
        v_cache: det(A_T * A_KV_ROWS, 10, HIDDEN_AMPLITUDE),
        inv_freq: (0..A_HEAD_DIM / 2)
            .map(|i| A_THETA.powf(-2.0 * i as f64 / A_HEAD_DIM as f64) as f32)
            .collect(),
    }
}

/// Which judged arms the lowered attention runs with.
#[derive(Clone, Copy)]
struct AttnArms {
    qk_norm: bool,
    v_norm: bool,
    /// Bind the K matrix as `v` too (Gemma 4's K≡V sliding layers).
    v_from_k: bool,
}

fn attn_geometry(fx: &AttnFixture) -> AttnGeometry<'_> {
    AttnGeometry {
        hidden: HIDDEN,
        num_q: A_NUM_Q,
        num_kv: A_NUM_KV,
        head_dim: A_HEAD_DIM,
        t: A_T,
        pos: A_POS,
        eps: EPS,
        norm_offset: RAW_OFFSET,
        qk_eps: EPS,
        qk_offset: RAW_OFFSET,
        score_scale: 1.0 / (A_HEAD_DIM as f32).sqrt(),
        inv_freq: &fx.inv_freq,
    }
}

fn attn_reference(fx: &AttnFixture, arms: AttnArms, v_source: VSource) -> Vec<f32> {
    cpu_attention(
        &fx.h,
        &attn_geometry(fx),
        &AttnOperands {
            norm_w: &fx.norm_w,
            wq: &fx.q.0,
            wk: &fx.k.0,
            wv: &fx.v.0,
            wo: &fx.o.0,
            k_cache: &fx.k_cache,
            v_cache: &fx.v_cache,
            q_norm: arms.qk_norm.then_some(fx.q_norm.as_slice()),
            k_norm: arms.qk_norm.then_some(fx.k_norm.as_slice()),
            v_norm: arms.v_norm,
            v_source,
        },
    )
}

fn run_attention(gpu: &MetalBackend, fx: &AttnFixture, arms: AttnArms) -> Vec<f32> {
    let h_in = gpu.lowering_upload(&fx.h).unwrap();
    let norm_w = gpu.lowering_upload(&fx.norm_w).unwrap();
    let q_norm = gpu.lowering_upload(&fx.q_norm).unwrap();
    let k_norm = gpu.lowering_upload(&fx.k_norm).unwrap();
    let k_cache = gpu.lowering_upload(&fx.k_cache).unwrap();
    let v_cache = gpu.lowering_upload(&fx.v_cache).unwrap();
    let inv_freq = gpu.lowering_upload(&fx.inv_freq).unwrap();
    let [qb, kb, vb, ob] = [&fx.q.1, &fx.k.1, &fx.v.1, &fx.o.1].map(|b| gpu.lowering_weight(b));
    let h_out = gpu.lowering_scratch(HIDDEN);
    let normed = gpu.lowering_scratch(HIDDEN);
    let attn_out = gpu.lowering_scratch(HIDDEN);
    let [q, gate, concat, gated] = [(); 4].map(|_| gpu.lowering_scratch(A_Q_ROWS));

    let w = AttnWeights {
        q: LoweredMatrix::F16 { bytes: &qb },
        k: LoweredMatrix::F16 { bytes: &kb },
        v: LoweredMatrix::F16 {
            bytes: if arms.v_from_k { &kb } else { &vb },
        },
        o: LoweredMatrix::F16 { bytes: &ob },
        gate: None,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        sinks: None,
        qk_norm: arms.qk_norm.then_some(QkNormWeights {
            q: &q_norm,
            k: &k_norm,
            weight_offset: RAW_OFFSET,
        }),
        norm_weight: &norm_w,
        post_norm: None,
    };
    let s = AttnScratch {
        normed: &normed,
        q: &q,
        k_cache: &k_cache,
        v_cache: &v_cache,
        gate: &gate,
        concat: &concat,
        gated: &gated,
        attn_out: &attn_out,
        inv_freq: &inv_freq,
    };
    let shape = AttnShape {
        hidden: HIDDEN,
        num_q_heads: A_NUM_Q,
        num_kv_heads: A_NUM_KV,
        head_dim: A_HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: RAW_OFFSET,
        qk_norm_eps: EPS,
        parameter_free_q: false,
        parameter_free_k: false,
        parameter_free_v: arms.v_norm,
        query_scale: None,
        score_scale: 1.0 / (A_HEAD_DIM as f32).sqrt(),
        position: LoweredPosition::Rope { theta: A_THETA },
        window: None,
        softcap: None,
        position_index: A_POS,
        kv_len: A_T,
    };
    run_once(gpu, |enc| {
        gpu.encode_attention(enc, &h_in, &h_out, &w, &s, &shape)
    });
    gpu.lowering_readback(&h_out, HIDDEN).unwrap()
}

/// `qk_norm: Some` lowers Gemma's weighted per-head Q/K norm at parity
/// with the HF order (projection → norm → rope); the same op absent
/// moves the output far beyond the parity residual, both on the CPU
/// reference and on the device.
#[test]
fn weighted_qk_norm_lowers_at_parity_and_is_load_bearing() {
    let Some(gpu) = device() else { return };
    let fx = attn_fixture();
    let on = AttnArms {
        qk_norm: true,
        v_norm: false,
        v_from_k: false,
    };
    let off = AttnArms {
        qk_norm: false,
        ..on
    };
    let got = run_attention(&gpu, &fx, on);
    let parity = assert_parity(
        "weighted qk norm",
        &attn_reference(&fx, on, VSource::Projection),
        &got,
        FRAGMENT_PARITY,
    );
    let want_off = attn_reference(&fx, off, VSource::Projection);
    assert_control(
        "qk norm absent (reference)",
        parity,
        rel_rms(&want_off, &got),
    );
    let got_off = run_attention(&gpu, &fx, off);
    assert_parity("qk norm absent", &want_off, &got_off, FRAGMENT_PARITY);
    assert_control("qk norm absent (device)", parity, rel_rms(&got_off, &got));
}

/// `parameter_free_v: true` lowers Gemma 4's weightless per-head RMS on
/// the raw value projection at parity; absent, the output moves.
#[test]
fn parameter_free_v_norm_lowers_at_parity_and_is_load_bearing() {
    let Some(gpu) = device() else { return };
    let fx = attn_fixture();
    let on = AttnArms {
        qk_norm: false,
        v_norm: true,
        v_from_k: false,
    };
    let off = AttnArms {
        v_norm: false,
        ..on
    };
    let got = run_attention(&gpu, &fx, on);
    let parity = assert_parity(
        "parameter-free v norm",
        &attn_reference(&fx, on, VSource::Projection),
        &got,
        FRAGMENT_PARITY,
    );
    let want_off = attn_reference(&fx, off, VSource::Projection);
    assert_control(
        "v norm absent (reference)",
        parity,
        rel_rms(&want_off, &got),
    );
    let got_off = run_attention(&gpu, &fx, off);
    assert_parity("v norm absent", &want_off, &got_off, FRAGMENT_PARITY);
    assert_control("v norm absent (device)", parity, rel_rms(&got_off, &got));
}

/// On a K≡V layer (the same matrix bound as `k` and `v`, weighted k-norm
/// and parameter-free v-norm both on) the lowering takes V from the RAW
/// K projection and v-norms it; the two wrong orders — V from the
/// weighted-normed K, V from the rotated K — are distinguishable.
#[test]
fn k_equals_v_binding_takes_v_from_the_raw_k_projection_then_v_norms_it() {
    let Some(gpu) = device() else { return };
    let fx = attn_fixture();
    let arms = AttnArms {
        qk_norm: true,
        v_norm: true,
        v_from_k: true,
    };
    let got = run_attention(&gpu, &fx, arms);
    let parity = assert_parity(
        "K≡V binding, V from raw K",
        &attn_reference(&fx, arms, VSource::RawK),
        &got,
        FRAGMENT_PARITY,
    );
    let normed_k = attn_reference(&fx, arms, VSource::NormedK);
    assert_control("V from k-normed K", parity, rel_rms(&normed_k, &got));
    let roped_k = attn_reference(&fx, arms, VSource::RopedK);
    assert_control("V from rotated K", parity, rel_rms(&roped_k, &got));
    let own_v = attn_reference(&fx, arms, VSource::Projection);
    assert_control("V from its own projection", parity, rel_rms(&own_v, &got));
}

// ── 4. tanh-GELU FFN ────────────────────────────────────────────────

const F_INTER: usize = 96;

fn run_ffn(
    gpu: &MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    w: &[&[u8]; 3],
    act: FfnActivation,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(h).unwrap();
    let norm = gpu.lowering_upload(norm_w).unwrap();
    let [gate_b, up_b, down_b] = w.map(|b| gpu.lowering_weight(b));
    let h_out = gpu.lowering_scratch(HIDDEN);
    let [normed, down] = [(); 2].map(|_| gpu.lowering_scratch(HIDDEN));
    let [gate, up, act_buf] = [(); 3].map(|_| gpu.lowering_scratch(F_INTER));
    let weights = FfnWeights {
        gate: LoweredMatrix::F16 { bytes: &gate_b },
        up: LoweredMatrix::F16 { bytes: &up_b },
        down: LoweredMatrix::F16 { bytes: &down_b },
        norm_weight: &norm,
        post_norm: None,
    };
    let scratch = FfnScratch {
        normed: &normed,
        gate: &gate,
        up: &up,
        act: &act_buf,
        down: &down,
    };
    let shape = FfnShape {
        hidden: HIDDEN,
        intermediate: F_INTER,
        norm_eps: EPS,
        norm_weight_offset: RAW_OFFSET,
        activation: act,
    };
    run_once(gpu, |enc| {
        gpu.encode_gated_ffn(enc, &h_in, &h_out, &weights, &scratch, &shape)
    });
    gpu.lowering_readback(&h_out, HIDDEN).unwrap()
}

/// `FfnActivation::GeluTanh` lowers `h + down(gelu_tanh(gate x) ⊙ up x)`
/// at parity with the crate's own `gelu_tanh`; the SiLU arm on the same
/// weights is at parity with ITS reference and the two are
/// distinguishable — so the activation is selected, not defaulted.
#[test]
fn gelu_tanh_ffn_lowers_at_parity_and_is_distinguishable_from_silu() {
    let Some(gpu) = device() else { return };
    let h = det(HIDDEN, 21, HIDDEN_AMPLITUDE);
    let norm_w = near_one(HIDDEN, 22, NORM_WEIGHT_AMPLITUDE);
    let gate = f16_matrix(&det(F_INTER * HIDDEN, 23, WEIGHT_AMPLITUDE));
    let up = f16_matrix(&det(F_INTER * HIDDEN, 24, WEIGHT_AMPLITUDE));
    let down = f16_matrix(&det(HIDDEN * F_INTER, 25, WEIGHT_AMPLITUDE));
    let reference = |gelu: bool| -> Vec<f32> {
        let branch = gated_ffn_branch(
            &h,
            &DenseFfn {
                norm_w: &norm_w,
                gate: &gate.0,
                up: &up.0,
                down: &down.0,
                hidden: HIDDEN,
                inter: F_INTER,
                eps: EPS,
                offset: RAW_OFFSET,
            },
            gelu,
        );
        h.iter().zip(&branch).map(|(a, b)| a + b).collect()
    };
    let bytes = [gate.1.as_slice(), up.1.as_slice(), down.1.as_slice()];
    let got_gelu = run_ffn(&gpu, &h, &norm_w, &bytes, FfnActivation::GeluTanh);
    let want_gelu = reference(true);
    let parity = assert_parity("gelu_tanh ffn", &want_gelu, &got_gelu, FRAGMENT_PARITY);
    let want_silu = reference(false);
    assert_control(
        "silu reference vs gelu device",
        parity,
        rel_rms(&want_silu, &got_gelu),
    );
    let got_silu = run_ffn(&gpu, &h, &norm_w, &bytes, FfnActivation::Silu);
    assert_parity("silu ffn", &want_silu, &got_silu, FRAGMENT_PARITY);
    assert_control(
        "silu vs gelu device arms",
        parity,
        rel_rms(&got_silu, &got_gelu),
    );
}

// ── 5. hybrid stack ─────────────────────────────────────────────────

/// Encode the whole hybrid stack once, entered from `h_a`, checkpoint
/// after every layer, and return the per-layer captures.
fn gpu_stack(gpu: &MetalBackend, h0: &[f32], fx: &[StackLayer]) -> Vec<Vec<f32>> {
    let dev: Vec<LayerDevice> = fx
        .iter()
        .enumerate()
        .map(|(i, l)| upload_layer(gpu, i, l))
        .collect();
    let h_a = gpu.lowering_upload(h0).unwrap();
    let zero = gpu.lowering_upload(&vec![0.0f32; HIDDEN]).unwrap();
    let hidden_bufs: Vec<metal::Buffer> = (0..16).map(|_| gpu.lowering_scratch(HIDDEN)).collect();
    let q_bufs: Vec<metal::Buffer> = (0..4).map(|_| gpu.lowering_scratch(S_Q_ROWS)).collect();
    let inter_bufs: Vec<metal::Buffer> =
        (0..3).map(|_| gpu.lowering_scratch(DENSE_INTER)).collect();
    let scratch = StackScratch {
        h_a: &h_a,
        h_b: &hidden_bufs[0],
        attn_normed: &hidden_bufs[1],
        q: &q_bufs[0],
        gate: &q_bufs[1],
        concat: &q_bufs[2],
        gated: &q_bufs[3],
        attn_out: &hidden_bufs[2],
        attn_post: &hidden_bufs[3],
        ffn_normed: &hidden_bufs[4],
        ffn_gate: &inter_bufs[0],
        ffn_up: &inter_bufs[1],
        ffn_act: &inter_bufs[2],
        ffn_down: &hidden_bufs[5],
        ffn_post: &hidden_bufs[6],
        hybrid: Some(HybridScratch {
            dense_out: &hidden_bufs[7],
            router_in: &hidden_bufs[8],
            expert_sum: &hidden_bufs[9],
            experts_out: &hidden_bufs[10],
            branch_sum: &hidden_bufs[11],
            zero: &zero,
        }),
    };
    let post_scratch = &hidden_bufs[12];
    let caps: Vec<metal::Buffer> = (0..LAYERS).map(|_| gpu.lowering_scratch(HIDDEN)).collect();
    let cps: Vec<Checkpoint> = caps
        .iter()
        .enumerate()
        .map(|(l, b)| Checkpoint {
            after_layer: l,
            into: b,
        })
        .collect();

    let layers: Vec<LayerLowering> = dev
        .iter()
        .zip(fx)
        .map(|(d, l)| LayerLowering {
            attn: AttnWeights {
                q: LoweredMatrix::F16 { bytes: &d.f16[0] },
                k: LoweredMatrix::F16 { bytes: &d.f16[1] },
                v: LoweredMatrix::F16 { bytes: &d.f16[2] },
                o: LoweredMatrix::F16 { bytes: &d.f16[3] },
                gate: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                o_bias: None,
                sinks: None,
                qk_norm: Some(QkNormWeights {
                    q: &d.f32[Q_NORM],
                    k: &d.f32[K_NORM],
                    weight_offset: RAW_OFFSET,
                }),
                norm_weight: &d.f32[ATTN_NORM],
                post_norm: None,
            },
            attn_shape: l.attn_shape(),
            ffn: LayerFfnLowering::Hybrid(Box::new(HybridFfnLowering {
                dense: FfnWeights {
                    gate: LoweredMatrix::F16 { bytes: &d.f16[4] },
                    up: LoweredMatrix::F16 { bytes: &d.f16[5] },
                    down: LoweredMatrix::F16 { bytes: &d.f16[6] },
                    norm_weight: &d.f32[PRE_FFN],
                    post_norm: None,
                },
                dense_shape: FfnShape {
                    hidden: HIDDEN,
                    intermediate: DENSE_INTER,
                    norm_eps: EPS,
                    norm_weight_offset: CENTRED_OFFSET,
                    activation: FfnActivation::GeluTanh,
                },
                routed: RoutedFfnLowering {
                    moe: l.moe(),
                    scratch: &d.scratch,
                    table: &d.table,
                    eps: EPS,
                },
                router_conditioning: &d.f32[ROUTER_COND],
                per_expert_scale: &d.f32[PE_SCALE],
                pre_experts_norm: &d.f32[PRE_EXPERTS],
                post_dense_norm: &d.f32[POST_DENSE],
                post_experts_norm: &d.f32[POST_EXPERTS],
                branch_norm_eps: EPS,
                branch_norm_weight_offset: CENTRED_OFFSET,
                post_ffn_norm: Some(PostNorm {
                    weight: &d.f32[POST_FFN],
                    eps: POST_EPS,
                    weight_offset: CENTRED_OFFSET,
                    scratch: post_scratch,
                }),
                layer_scale: Some(l.layer_scale),
            })),
            k_cache: &d.f32[K_CACHE],
            v_cache: &d.f32[V_CACHE],
            inv_freq: &d.f32[INV_FREQ],
        })
        .collect();

    let mut final_buf: Option<&metal::Buffer> = None;
    run_once(gpu, |enc| {
        final_buf = Some(gpu.encode_stack(enc, &h_a, &layers, &scratch, &cps));
    });
    assert!(
        std::ptr::eq(final_buf.unwrap(), &h_a),
        "stack entered from h_a must finish in h_a after {LAYERS} layers"
    );
    caps.iter()
        .map(|b| gpu.lowering_readback(b, HIDDEN).unwrap())
        .collect()
}

/// A two-layer hybrid stack — weighted QK norm + V norm + per-layer rope
/// tables of different width in attention, dense tanh-GELU branch plus
/// routed MXFP4 experts with router conditioning, renormalised top-k,
/// per-expert scale, the three branch norms, the post-FFN norm and the
/// layer scale — matches the CPU reference at both checkpoints, and the
/// per-expert scale and layer scale are each live.
#[test]
fn hybrid_stack_checkpoints_match_cpu_reference_and_scales_are_live() {
    let Some(gpu) = device() else { return };
    let h0 = det(HIDDEN, 777, HIDDEN_AMPLITUDE);
    let fx: Vec<StackLayer> = (0..LAYERS).map(|l| build_layer(l, Tweak::None)).collect();
    let want = cpu_stack(&h0, &fx);
    let got = gpu_stack(&gpu, &h0, &fx);
    let mut worst = 0.0f64;
    for (l, (w, g)) in want.iter().zip(&got).enumerate() {
        worst = worst.max(assert_parity(
            &format!("hybrid stack checkpoint after layer {l}"),
            w,
            g,
            STACK_PARITY,
        ));
    }

    // Control 1: double the per-expert scale of layer 0's top-1 expert
    // (known from the CPU route). `fx` stays alive so the descriptor
    // cache cannot conflate the two layer-0 banks.
    let victim = hybrid_route(&fx[0].cpu_attention(&h0), &fx[0].hybrid_ref())[0].0;
    let scaled = vec![
        build_layer(0, Tweak::PerExpertScale(victim)),
        build_layer(1, Tweak::None),
    ];
    let got_scaled = gpu_stack(&gpu, &h0, &scaled);
    for l in 0..LAYERS {
        assert_control(
            &format!("per-expert scale of expert {victim}, checkpoint {l}"),
            worst,
            rel_rms(&got[l], &got_scaled[l]),
        );
    }
    let want_scaled = cpu_stack(&h0, &scaled);
    assert_parity(
        "perturbed per-expert scale, layer 0",
        &want_scaled[0],
        &got_scaled[0],
        STACK_PARITY,
    );

    // Control 2: the layer scale of layer 0.
    let rescaled = vec![
        build_layer(0, Tweak::LayerScale),
        build_layer(1, Tweak::None),
    ];
    let got_rescaled = gpu_stack(&gpu, &h0, &rescaled);
    for l in 0..LAYERS {
        assert_control(
            &format!("layer scale, checkpoint {l}"),
            worst,
            rel_rms(&got[l], &got_rescaled[l]),
        );
    }
    let want_rescaled = cpu_stack(&h0, &rescaled);
    assert_parity(
        "perturbed layer scale, layer 0",
        &want_rescaled[0],
        &got_rescaled[0],
        STACK_PARITY,
    );
}
