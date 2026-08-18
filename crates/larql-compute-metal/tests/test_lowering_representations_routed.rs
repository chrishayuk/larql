//! The lowering's representation dispatch and its routed-FFN seam,
//! isolated from the CLI (A-9): the arms `larql vindex3 exec
//! --backend metal-lowered` exercises on gpt-oss-20b, on a synthetic
//! fixture small enough to judge against a CPU reference.
//!
//! | proof | what it establishes |
//! |---|---|
//! | matvec arms | `encode_matvec` selects the F16 and MXFP4 kernels, each at parity with its own decoded reference, and the two arms are distinguishable |
//! | region registration | page-aligned bytes register; a misaligned view of the same allocation refuses |
//! | routed stack | a two-layer stack with routed FFNs, hidden state resident and ping-ponging through both scratch slots, checkpoints at parity with the CPU reference; the expert bytes are live |
//! | descriptor refusal | operands outside a registered region yield `None`, never a fallback |
//!
//! ## Why the fixture is small and awkward
//!
//! MXFP4 fixes `k ≡ 0 mod 32`; everything else is chosen to leave the
//! kernels' tiling non-trivial (matvec rows not a multiple of the
//! rows-per-threadgroup, four experts of which two are routed, two
//! layers so the ping-pong buffers actually alternate). The stack is
//! entered with `h_in = h_a` on purpose: that is the branch of the
//! ping-pong that a stack entered from a foreign upload never takes.

#![cfg(target_os = "macos")]

#[path = "lowering_routed_support/mod.rs"]
mod support;

use larql_compute::{
    MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::lowering::attention::{AttnShape, AttnWeights, LoweredPosition};
use larql_compute_metal::lowering::stack::{
    Checkpoint, LayerFfnLowering, LayerLowering, RoutedFfnLowering, StackScratch,
};
use larql_compute_metal::lowering::{LoweredMatrix, MatvecTarget};
use larql_compute_metal::moe_descriptor::MoeExpertDescriptorTable;
use larql_compute_metal::{MetalBackend, MoeScratch};
use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
use support::{
    det, f16_matrix, matvec, rel_rms, rms_norm, routed_ffn_reference, AlignedRegion, Mxfp4Matrix,
    RoutedRef,
};

// ── geometry ─────────────────────────────────────────────────────────

const HIDDEN: usize = 64;
const INTER: usize = 64;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const LAYERS: usize = 2;
const NUM_Q: usize = 2;
const NUM_KV: usize = 1;
const HEAD_DIM: usize = 32;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
/// Cache length including the decoded position.
const T: usize = 4;
const POS: usize = T - 1;
/// Matvec rows for the arm test — deliberately not a multiple of the
/// kernels' rows-per-threadgroup, so the tail tile is exercised.
const MATVEC_ROWS: usize = 50;
const MATVEC_K: usize = 2 * MXFP4_GROUP_ELEMS;

// ── semantics (gpt-oss shaped) ───────────────────────────────────────

const EPS: f32 = 1e-5;
/// gpt-oss norms multiply by the raw weight (no centred `1 + w`).
const NORM_OFFSET: f32 = 0.0;
const GATE_RULE: MoeGateRule = MoeGateRule::ClampedGlu {
    limit: 7.0,
    alpha: 1.702,
};
const WEIGHT_AMPLITUDE: f32 = 0.35;
const BIAS_AMPLITUDE: f32 = 0.2;
const HIDDEN_AMPLITUDE: f32 = 1.0;
/// Norm weights sit near one so the routed layer's activations stay in
/// range for a 4-bit expert.
const NORM_WEIGHT_AMPLITUDE: f32 = 0.2;

// ── tolerances ───────────────────────────────────────────────────────

/// Both matvec arms are exact-grid references (f16 round-trip / MXFP4
/// decode); the only slack is f32 reassociation on the GPU.
const MATVEC_PARITY: f64 = 1e-4;
/// The two representations of one matrix must be *distinguishable*, or
/// the arm test could pass with either kernel bound to either variant.
/// 4-bit noise on a dense product is far above this.
const ARM_DISTINCTION_MIN: f64 = 1e-2;
/// Two composed layers, gpt-oss routing, f32 both sides.
const LAYER_PARITY: f64 = 1e-3;
/// A perturbed expert must move the checkpoint by well over the parity
/// bar, or the parity gate is blind to the expert bytes.
const CONTROL_MIN: f64 = 1e-2;

fn device() -> Option<MetalBackend> {
    let gpu = MetalBackend::new();
    if gpu.is_none() {
        eprintln!("no Metal device; skipping");
    }
    gpu
}

/// Run one encoder-full of work in one command buffer and wait.
fn run_once(gpu: &MetalBackend, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

// ── 1. encode_matvec representation arms ────────────────────────────

/// `encode_matvec` binds the f16 gemv for `LoweredMatrix::F16` and the
/// MXFP4 matvec for `LoweredMatrix::Mxfp4`; each lands at parity with
/// its own decoded reference, and on the same source matrix the two arms
/// disagree by 4-bit noise — so the arm genuinely selected the format.
#[test]
fn encode_matvec_selects_f16_and_mxfp4_arms_each_at_parity_with_its_decoder() {
    let Some(gpu) = device() else { return };
    let w = det(MATVEC_ROWS * MATVEC_K, 11, WEIGHT_AMPLITUDE);
    let x = det(MATVEC_K, 12, HIDDEN_AMPLITUDE);
    let (w_f16, f16_bytes) = f16_matrix(&w);
    let mx = Mxfp4Matrix::quantize(&w, MATVEC_ROWS, MATVEC_K);
    let w_mx = mx.dequantized();
    let want_f16 = matvec(&w_f16, &x, MATVEC_ROWS, MATVEC_K);
    let want_mx = matvec(&w_mx, &x, MATVEC_ROWS, MATVEC_K);

    let f16_buf = gpu.lowering_weight(&f16_bytes);
    let packed = gpu.lowering_weight(&mx.packed);
    let scales = gpu.lowering_weight(&mx.scales);
    let x_buf = gpu.lowering_upload(&x).unwrap();
    let out_f16 = gpu.lowering_scratch(MATVEC_ROWS);
    let out_mx = gpu.lowering_scratch(MATVEC_ROWS);

    run_once(&gpu, |enc| {
        gpu.encode_matvec(
            enc,
            &LoweredMatrix::F16 { bytes: &f16_buf },
            &MatvecTarget {
                x: &x_buf,
                out: &out_f16,
                out_offset: 0,
                n: MATVEC_ROWS,
                k: MATVEC_K,
            },
        );
        gpu.encode_matvec(
            enc,
            &LoweredMatrix::Mxfp4 {
                packed: &packed,
                scales: &scales,
            },
            &MatvecTarget {
                x: &x_buf,
                out: &out_mx,
                out_offset: 0,
                n: MATVEC_ROWS,
                k: MATVEC_K,
            },
        );
    });
    let got_f16 = gpu.lowering_readback(&out_f16, MATVEC_ROWS).unwrap();
    let got_mx = gpu.lowering_readback(&out_mx, MATVEC_ROWS).unwrap();

    let e_f16 = rel_rms(&want_f16, &got_f16);
    let e_mx = rel_rms(&want_mx, &got_mx);
    eprintln!("f16 arm rel_rms {e_f16:.3e}; mxfp4 arm rel_rms {e_mx:.3e}");
    assert!(
        e_f16 < MATVEC_PARITY,
        "F16 arm off its reference: {e_f16:.3e}"
    );
    assert!(
        e_mx < MATVEC_PARITY,
        "MXFP4 arm off its reference: {e_mx:.3e}"
    );

    // Control: the arms are distinguishable on the same W.
    let between = rel_rms(&got_f16, &got_mx);
    eprintln!("f16 vs mxfp4 arms on the same W: rel_rms {between:.3e}");
    assert!(
        between > ARM_DISTINCTION_MIN,
        "arm control BLIND: f16 and mxfp4 outputs agree to {between:.3e}"
    );
}

// ── 2. region registration ──────────────────────────────────────────

/// `lowering_register_region` accepts a page-aligned region and refuses
/// a view of the same allocation that starts one byte in.
#[test]
fn register_region_accepts_page_aligned_and_refuses_misaligned_bytes() {
    let Some(gpu) = device() else { return };
    let region = AlignedRegion::from_bytes(&[0xA5u8; 3 * MXFP4_GROUP_BYTES]);
    assert!(
        gpu.lowering_register_region(region.as_slice()),
        "page-aligned region refused"
    );
    assert!(
        gpu.lowering_register_region(region.as_slice()),
        "re-registering the same region is idempotent"
    );
    assert!(
        !gpu.lowering_register_region(region.misaligned_slice()),
        "a misaligned view registered — zero-copy binding would read the wrong bytes"
    );
}

// ── 3. routed layers through the stack ──────────────────────────────

/// One layer's host-side fixture: attention weights as f16 (the F16 arm
/// in situ), a routed FFN with MXFP4 experts in registered regions.
struct LayerFixture {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_bytes: Vec<u8>,
    k_bytes: Vec<u8>,
    v_bytes: Vec<u8>,
    o_bytes: Vec<u8>,
    attn_norm: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    gu_payload: AlignedRegion,
    gu_scales: AlignedRegion,
    dn_payload: AlignedRegion,
    dn_scales: AlignedRegion,
    router_proj: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
    pre_norm: Vec<f32>,
    reference: RoutedRef,
}

const GU_ROWS: usize = 2 * INTER;
const GU_PAYLOAD_PER_EXPERT: usize = GU_ROWS * (HIDDEN / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
const GU_SCALES_PER_EXPERT: usize = GU_ROWS * (HIDDEN / MXFP4_GROUP_ELEMS);
const DN_PAYLOAD_PER_EXPERT: usize = HIDDEN * (INTER / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
const DN_SCALES_PER_EXPERT: usize = HIDDEN * (INTER / MXFP4_GROUP_ELEMS);
/// Byte mask the down-perturbation control XORs into one expert's
/// packed nibbles — flips every code, so a live read must move.
const PERTURB_MASK: u8 = 0x77;
const SEED_STRIDE: u32 = 101;

fn build_layer(l: u32, perturb_down_expert: Option<usize>) -> LayerFixture {
    let s = |i: u32| l * SEED_STRIDE + i;
    let mk16 = |n: usize, k: usize, seed: u32| f16_matrix(&det(n * k, seed, WEIGHT_AMPLITUDE));
    let (q, q_bytes) = mk16(Q_ROWS, HIDDEN, s(1));
    let (k, k_bytes) = mk16(KV_ROWS, HIDDEN, s(2));
    let (v, v_bytes) = mk16(KV_ROWS, HIDDEN, s(3));
    let (o, o_bytes) = mk16(HIDDEN, Q_ROWS, s(4));
    let near_one = |seed: u32| -> Vec<f32> {
        det(HIDDEN, seed, NORM_WEIGHT_AMPLITUDE)
            .into_iter()
            .map(|w| 1.0 + w)
            .collect()
    };

    let mut gu_payload = Vec::new();
    let mut gu_scales = Vec::new();
    let mut dn_payload = Vec::new();
    let mut dn_scales = Vec::new();
    let mut gate_up = Vec::new();
    let mut down = Vec::new();
    for e in 0..NUM_EXPERTS as u32 {
        let gu = Mxfp4Matrix::quantize(
            &det(GU_ROWS * HIDDEN, s(20 + 2 * e), WEIGHT_AMPLITUDE),
            GU_ROWS,
            HIDDEN,
        );
        let mut dn = Mxfp4Matrix::quantize(
            &det(HIDDEN * INTER, s(21 + 2 * e), WEIGHT_AMPLITUDE),
            HIDDEN,
            INTER,
        );
        if perturb_down_expert == Some(e as usize) {
            dn.packed.iter_mut().for_each(|b| *b ^= PERTURB_MASK);
        }
        gate_up.push(gu.dequantized());
        down.push(dn.dequantized());
        gu_payload.extend_from_slice(&gu.packed);
        gu_scales.extend_from_slice(&gu.scales);
        dn_payload.extend_from_slice(&dn.packed);
        dn_scales.extend_from_slice(&dn.scales);
    }
    assert_eq!(gu_payload.len(), NUM_EXPERTS * GU_PAYLOAD_PER_EXPERT);
    assert_eq!(dn_payload.len(), NUM_EXPERTS * DN_PAYLOAD_PER_EXPERT);

    LayerFixture {
        q,
        k,
        v,
        o,
        q_bytes,
        k_bytes,
        v_bytes,
        o_bytes,
        attn_norm: near_one(s(5)),
        k_cache: det(T * KV_ROWS, s(6), HIDDEN_AMPLITUDE),
        v_cache: det(T * KV_ROWS, s(7), HIDDEN_AMPLITUDE),
        gu_payload: AlignedRegion::from_bytes(&gu_payload),
        gu_scales: AlignedRegion::from_bytes(&gu_scales),
        dn_payload: AlignedRegion::from_bytes(&dn_payload),
        dn_scales: AlignedRegion::from_bytes(&dn_scales),
        router_proj: det(NUM_EXPERTS * HIDDEN, s(8), WEIGHT_AMPLITUDE),
        router_bias: det(NUM_EXPERTS, s(9), BIAS_AMPLITUDE),
        gate_up_bias: det(NUM_EXPERTS * GU_ROWS, s(10), BIAS_AMPLITUDE),
        down_bias: det(NUM_EXPERTS * HIDDEN, s(11), BIAS_AMPLITUDE),
        pre_norm: near_one(s(12)),
        reference: RoutedRef {
            gate_up,
            down,
            hidden: HIDDEN,
            inter: INTER,
        },
    }
}

/// Per-expert byte slices into a bank of `NUM_EXPERTS` equal `per`-byte
/// slices.
fn expert_slices(bank: &[u8], per: usize) -> Vec<&[u8]> {
    (0..NUM_EXPERTS)
        .map(|e| &bank[e * per..(e + 1) * per])
        .collect()
}

impl LayerFixture {
    /// The `MoeLayerWeights` view over the registered banks — the same
    /// shape the CLI's `RoutedLayer::moe` builds from a `RoutedFfnOp`.
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            experts_gate_up: expert_slices(self.gu_payload.as_slice(), GU_PAYLOAD_PER_EXPERT),
            experts_down: expert_slices(self.dn_payload.as_slice(), DN_PAYLOAD_PER_EXPERT),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(self.gu_scales.as_slice(), GU_SCALES_PER_EXPERT),
                down: expert_slices(self.dn_scales.as_slice(), DN_SCALES_PER_EXPERT),
            },
            fused_row_layout: MoeFusedRowLayout::Interleaved,
            expert_data_format: QuantFormat::MXFP4,
            router_proj: &self.router_proj,
            router_bias: &self.router_bias,
            experts_gate_up_bias: &self.gate_up_bias,
            experts_down_bias: &self.down_bias,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &self.pre_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTER,
            gate_rule: GATE_RULE,
        }
    }

    fn register(&self, gpu: &MetalBackend) {
        for bank in [
            &self.gu_payload,
            &self.gu_scales,
            &self.dn_payload,
            &self.dn_scales,
        ] {
            assert!(gpu.lowering_register_region(bank.as_slice()));
        }
    }
}

/// CPU attention for one layer (NoPE, no gate, no post-norm, full
/// span): `h + Wo · attend(Wq n, cache ∪ Wk n, cache ∪ Wv n)`.
fn cpu_attention(h: &[f32], w: &LayerFixture) -> Vec<f32> {
    let normed = rms_norm(h, &w.attn_norm, EPS, NORM_OFFSET);
    let q = matvec(&w.q, &normed, Q_ROWS, HIDDEN);
    let k = matvec(&w.k, &normed, KV_ROWS, HIDDEN);
    let v = matvec(&w.v, &normed, KV_ROWS, HIDDEN);
    let mut kc = w.k_cache.clone();
    let mut vc = w.v_cache.clone();
    kc[POS * KV_ROWS..].copy_from_slice(&k);
    vc[POS * KV_ROWS..].copy_from_slice(&v);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut concat = vec![0.0f32; Q_ROWS];
    for head in 0..NUM_Q {
        let kv = head / (NUM_Q / NUM_KV);
        let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let sc: Vec<f32> = (0..T)
            .map(|t| {
                let kh = &kc[t * KV_ROWS + kv * HEAD_DIM..t * KV_ROWS + (kv + 1) * HEAD_DIM];
                qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
        let den: f32 = ex.iter().sum();
        for d in 0..HEAD_DIM {
            concat[head * HEAD_DIM + d] = (0..T)
                .map(|t| ex[t] / den * vc[t * KV_ROWS + kv * HEAD_DIM + d])
                .sum();
        }
    }
    let ao = matvec(&w.o, &concat, HIDDEN, Q_ROWS);
    h.iter().zip(&ao).map(|(a, b)| a + b).collect()
}

/// CPU: attention then the routed FFN, per layer, capturing every
/// layer's output.
fn cpu_stack(h0: &[f32], layers: &[LayerFixture]) -> Vec<Vec<f32>> {
    let mut h = h0.to_vec();
    let mut caps = Vec::new();
    for w in layers {
        let h1 = cpu_attention(&h, w);
        h = routed_ffn_reference(&h1, &w.moe(), &w.reference, EPS, NORM_OFFSET);
        caps.push(h.clone());
    }
    caps
}

/// Device-side state for one layer, kept alive for the whole stack.
struct LayerDevice {
    weights: [metal::Buffer; 4],
    attn_norm: metal::Buffer,
    k_cache: metal::Buffer,
    v_cache: metal::Buffer,
    scratch: MoeScratch,
    table: std::sync::Arc<MoeExpertDescriptorTable>,
}

fn upload_layer(gpu: &MetalBackend, idx: usize, w: &LayerFixture) -> LayerDevice {
    w.register(gpu);
    let moe = w.moe();
    let scratch =
        MoeScratch::new_public_with_format(gpu, TOP_K, HIDDEN, INTER, QuantFormat::MXFP4, HIDDEN);
    assert!(
        gpu.lowering_moe_supported(&moe, &scratch),
        "layer {idx}: descriptor MoE path refused a gpt-oss-shaped routed layer"
    );
    let table = gpu
        .lowering_moe_descriptor(idx, &moe, INTER, HIDDEN)
        .expect("expert slices lie in registered regions");
    LayerDevice {
        weights: [
            gpu.lowering_weight(&w.q_bytes),
            gpu.lowering_weight(&w.k_bytes),
            gpu.lowering_weight(&w.v_bytes),
            gpu.lowering_weight(&w.o_bytes),
        ],
        attn_norm: gpu.lowering_upload(&w.attn_norm).unwrap(),
        k_cache: gpu.lowering_upload(&w.k_cache).unwrap(),
        v_cache: gpu.lowering_upload(&w.v_cache).unwrap(),
        scratch,
        table,
    }
}

fn attn_shape() -> AttnShape {
    AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: NORM_OFFSET,
        qk_norm_eps: EPS,
        parameter_free_q: false,
        parameter_free_k: false,
        query_scale: None,
        score_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        position: LoweredPosition::None,
        window: None,
        softcap: None,
        position_index: POS,
        kv_len: T,
    }
}

/// Encode the whole stack once, entered from `h_a`, checkpoint after
/// every layer, and return the per-layer captures.
fn gpu_stack(gpu: &MetalBackend, h0: &[f32], fx: &[LayerFixture]) -> Vec<Vec<f32>> {
    let dev: Vec<LayerDevice> = fx
        .iter()
        .enumerate()
        .map(|(i, w)| upload_layer(gpu, i, w))
        .collect();

    // Enter the stack FROM h_a: the ping-pong then takes the h_b/h_a arm.
    let h_a = gpu.lowering_upload(h0).unwrap();
    let inv_freq = gpu.lowering_upload(&[0.0f32; HEAD_DIM / 2]).unwrap();
    let sc: Vec<metal::Buffer> = (0..13)
        .map(|i| match i {
            2..=4 => gpu.lowering_scratch(Q_ROWS),
            8..=10 => gpu.lowering_scratch(INTER),
            _ => gpu.lowering_scratch(HIDDEN),
        })
        .collect();
    let scratch = StackScratch {
        h_a: &h_a,
        h_b: &sc[0],
        attn_normed: &sc[1],
        q: &sc[2],
        gate: &sc[3],
        concat: &sc[4],
        gated: &sc[11],
        attn_out: &sc[5],
        attn_post: &sc[6],
        ffn_normed: &sc[7],
        ffn_gate: &sc[8],
        ffn_up: &sc[9],
        ffn_act: &sc[10],
        ffn_down: &sc[12],
        ffn_post: &sc[1],
        inv_freq: &inv_freq,
    };
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
        .map(|(d, w)| LayerLowering {
            attn: AttnWeights {
                q: LoweredMatrix::F16 {
                    bytes: &d.weights[0],
                },
                k: LoweredMatrix::F16 {
                    bytes: &d.weights[1],
                },
                v: LoweredMatrix::F16 {
                    bytes: &d.weights[2],
                },
                o: LoweredMatrix::F16 {
                    bytes: &d.weights[3],
                },
                gate: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                o_bias: None,
                sinks: None,
                norm_weight: &d.attn_norm,
                post_norm: None,
            },
            attn_shape: attn_shape(),
            ffn: LayerFfnLowering::Routed(Box::new(RoutedFfnLowering {
                moe: w.moe(),
                scratch: &d.scratch,
                table: &d.table,
                eps: EPS,
            })),
            k_cache: &d.k_cache,
            v_cache: &d.v_cache,
        })
        .collect();

    let mut final_buf: Option<&metal::Buffer> = None;
    run_once(gpu, |enc| {
        final_buf = Some(gpu.encode_stack(enc, &h_a, &layers, &scratch, &cps));
    });
    // Two layers entered from h_a: each layer writes mid = h_b, dst = h_a.
    assert!(
        std::ptr::eq(final_buf.unwrap(), &h_a),
        "stack entered from h_a must finish in h_a after {LAYERS} layers"
    );
    caps.iter()
        .map(|b| gpu.lowering_readback(b, HIDDEN).unwrap())
        .collect()
}

/// A two-layer stack whose FFNs are routed MXFP4 experts in registered
/// regions matches the CPU reference at every checkpoint (route, gate
/// rule, biases, pre-experts norm all judged); the residual ping-pongs
/// through both scratch slots; and the expert bytes are live — flipping
/// one selected expert's down nibbles moves the checkpoint.
#[test]
fn routed_stack_checkpoints_match_cpu_reference_and_expert_bytes_are_live() {
    let Some(gpu) = device() else { return };
    let h0 = det(HIDDEN, 999, HIDDEN_AMPLITUDE);
    let fx: Vec<LayerFixture> = (0..LAYERS as u32).map(|l| build_layer(l, None)).collect();
    let want = cpu_stack(&h0, &fx);
    let got = gpu_stack(&gpu, &h0, &fx);
    let mut worst = 0.0f64;
    for (l, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(g.iter().all(|v| v.is_finite()), "layer {l}: non-finite");
        let e = rel_rms(w, g);
        eprintln!("routed stack checkpoint after layer {l}: rel_rms {e:.3e}");
        assert!(e < LAYER_PARITY, "layer {l} diverges: {e:.3e}");
        worst = worst.max(e);
    }

    // Control: perturb the down bytes of layer 0's top-1 expert (known
    // from the CPU route) and re-run. Both fixtures stay alive so no
    // address-keyed cache can conflate them.
    let victim = {
        let x0 = layer0_router_input(&h0, &fx[0]);
        larql_compute::cpu::ops::moe::moe_route_from_router_input(&x0, &fx[0].moe()).0[0]
    };
    // The descriptor cache keys on (layer, bank address); the perturbed
    // bank is a fresh allocation and `fx` is still alive, so the two
    // layer-0 tables cannot collide.
    let perturbed: Vec<LayerFixture> = vec![build_layer(0, Some(victim)), build_layer(1, None)];
    let got_p = gpu_stack(&gpu, &h0, &perturbed);
    let moved = rel_rms(&got[0], &got_p[0]);
    eprintln!("down-byte perturbation moved checkpoint 0 by rel_rms {moved:.3e} (parity worst {worst:.3e})");
    assert!(
        moved > CONTROL_MIN && moved > worst * 10.0,
        "control BLIND: perturbing expert {victim}'s down bytes moved the output by only {moved:.3e}"
    );
    // And the perturbed stack still matches ITS OWN reference: the
    // reference tracks the bytes, not a fixed answer.
    let want_p = cpu_stack(&h0, &perturbed);
    let e = rel_rms(&want_p[0], &got_p[0]);
    assert!(
        e < LAYER_PARITY,
        "perturbed layer 0 off its reference: {e:.3e}"
    );
}

/// Layer 0's routed-FFN router input on the CPU — the post-attention
/// residual, pre-experts-normed.
fn layer0_router_input(h0: &[f32], w: &LayerFixture) -> Vec<f32> {
    rms_norm(&cpu_attention(h0, w), &w.pre_norm, EPS, NORM_OFFSET)
}

// ── 4. descriptor refusal ───────────────────────────────────────────

/// A routed layer whose expert slices are owned `Vec` bytes — nowhere
/// in a registered region — is *supported* by policy/format yet yields
/// no descriptor: the fail-closed contract, not a copying fallback.
#[test]
fn moe_descriptor_refuses_expert_slices_outside_registered_regions() {
    let Some(gpu) = device() else { return };
    let fx = build_layer(7, None);
    // Same bytes, owned copies: valid data, wrong residency.
    let gu_payload = fx.gu_payload.as_slice().to_vec();
    let gu_scales = fx.gu_scales.as_slice().to_vec();
    let dn_payload = fx.dn_payload.as_slice().to_vec();
    let dn_scales = fx.dn_scales.as_slice().to_vec();
    let moe = MoeLayerWeights {
        experts_gate_up: expert_slices(&gu_payload, GU_PAYLOAD_PER_EXPERT),
        experts_down: expert_slices(&dn_payload, DN_PAYLOAD_PER_EXPERT),
        expert_scales: MoeExpertScales::Paired {
            gate_up: expert_slices(&gu_scales, GU_SCALES_PER_EXPERT),
            down: expert_slices(&dn_scales, DN_SCALES_PER_EXPERT),
        },
        ..fx.moe()
    };
    let scratch =
        MoeScratch::new_public_with_format(&gpu, TOP_K, HIDDEN, INTER, QuantFormat::MXFP4, HIDDEN);
    assert!(
        gpu.lowering_moe_supported(&moe, &scratch),
        "policy/format support is independent of residency"
    );
    assert!(
        gpu.lowering_moe_descriptor(7, &moe, INTER, HIDDEN)
            .is_none(),
        "descriptor built over unregistered expert bytes — the lowering would bind copies"
    );
}
