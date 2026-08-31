//! The layer's NEW links, against a host reference.
//!
//! The attention is already gated in `trait_impl::kda::tests` and the
//! experts in `trait_impl::bf16_moe_block::tests`, so the reference here
//! composes those proven device calls and computes only what this module
//! adds — the norms, the residual, the router decision and the weighted
//! combine. That keeps the comparison pointed at the code under test
//! rather than re-deriving a whole decoder layer.

use super::*;

/// Reach the routed weights of a layer built by `layer_weights`, for
/// the controls that corrupt one field. Panics on a dense layer, which
/// no control here builds.
fn moe_mut<'a, 'b>(w: &'b mut KimiLayerWeights<'a>) -> &'b mut KimiMoeWeights<'a> {
    match &mut w.ffn {
        FfnSpec::Moe(m) => m,
        FfnSpec::Dense(_) => panic!("this control corrupts a routed layer"),
    }
}
use crate::trait_impl::bf16_moe_block::{BlockLowering, ExpertBankRef, MoeBlockCall, MoeFfnBanks};
use crate::trait_impl::grouped_experts::ExpertOffset;
use crate::trait_impl::kda::KdaDeviceState;
use crate::trait_impl::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use crate::MetalBackend;

const HEADS: usize = 2;
const DIM: usize = 4;
const HIDDEN: usize = 8;
const INTER: usize = 6;
const KERNEL: usize = 4;
const WIDTH: usize = HEADS * DIM;
const EXPERTS: usize = 12;
const TOP_K: usize = 3;
const RESIDENT: usize = 5;
const BRANCH_SCALE: f32 = 2.446;
const EPS: f32 = 1e-5;
const TOLERANCE: f32 = 1e-4;

fn shape() -> KdaShape {
    KdaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        head_dim: DIM,
        conv_kernel: KERNEL,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.41 + seed).sin() * 0.5)
        .collect()
}

fn bf16_bytes(n: usize, k: usize, seed: f32) -> Vec<u8> {
    synth(n * k, seed)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A layer's weights, owned. The resident bank holds `RESIDENT` routed
/// experts; the shared branch lives in its OWN allocations, one per
/// projection — semantic identity, not co-location. The router scores
/// all `EXPERTS`, so the residency table is what decides whether a
/// route is servable.
struct Fixture {
    x: Vec<f32>,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    router_weight: Vec<f32>,
    router_bias: Vec<f32>,
    residency: Vec<u32>,
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    o_proj: Vec<u8>,
    kda_f32: Vec<Vec<f32>>,
}

/// One projection's bank: a routed region, a table, and the shared
/// branch's own region.
fn projection<'a>(routed: &'a [u8], table: &'a [u32], shared: &'a [u8]) -> ProjectionBank<'a> {
    ProjectionBank {
        routed: EncodedRegion {
            bytes: routed,
            encoding: ExpertEncoding::Bf16,
        },
        addressing: ExpertAddressing::Table(table),
        shared: Some(EncodedRegion {
            bytes: shared,
            encoding: ExpertEncoding::Bf16,
        }),
    }
}

fn fixture() -> Fixture {
    let per_qkv = WIDTH * HIDDEN * 2;
    let mut qkv_bank = Vec::with_capacity(3 * per_qkv);
    for (i, seed) in [0.1f32, 1.3, 2.7].into_iter().enumerate() {
        let _ = i;
        qkv_bank.extend_from_slice(&bf16_bytes(WIDTH, HIDDEN, seed));
    }
    let gate_per = INTER * HIDDEN * 2;
    let down_per = HIDDEN * INTER * 2;
    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![layer_shader::NOT_RESIDENT; EXPERTS];
    // Bias the FIRST `RESIDENT` experts into residency, and bias the
    // router towards exactly those so the default route is servable.
    for (slot, entry) in residency.iter_mut().enumerate().take(RESIDENT) {
        *entry = (slot * gate_per) as u32;
        bank_gate.extend_from_slice(&bf16_bytes(INTER, HIDDEN, 3.0 + slot as f32));
        bank_up.extend_from_slice(&bf16_bytes(INTER, HIDDEN, 9.0 + slot as f32));
        bank_down.extend_from_slice(&bf16_bytes(HIDDEN, INTER, 15.0 + slot as f32));
    }
    debug_assert_eq!(bank_down.len(), RESIDENT * down_per);

    // A correction bias that puts the resident experts on top — the
    // point of the fixture is the seam, not a route that cannot be served.
    let router_bias: Vec<f32> = (0..EXPERTS)
        .map(|e| if e < RESIDENT { 1.0 } else { 0.0 })
        .collect();

    Fixture {
        x: synth(HIDDEN, 0.7),
        input_norm: synth(HIDDEN, 2.2).iter().map(|v| v + 1.0).collect(),
        post_norm: synth(HIDDEN, 3.3).iter().map(|v| v + 1.0).collect(),
        router_weight: synth(EXPERTS * HIDDEN, 4.4),
        router_bias,
        residency,
        bank_gate,
        bank_up,
        bank_down,
        shared_gate: bf16_bytes(INTER, HIDDEN, 21.0),
        shared_up: bf16_bytes(INTER, HIDDEN, 22.0),
        shared_down: bf16_bytes(HIDDEN, INTER, 23.0),
        qkv_bank,
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_qkv as u32),
            ExpertOffset((2 * per_qkv) as u32),
        ],
        o_proj: bf16_bytes(HIDDEN, WIDTH, 5.5),
        kda_f32: vec![
            synth(WIDTH * KERNEL, 0.5),                         // q_conv1d
            synth(WIDTH * KERNEL, 1.5),                         // k_conv1d
            synth(WIDTH * KERNEL, 2.5),                         // v_conv1d
            synth(DIM * HIDDEN, 6.1),                           // f_a
            synth(WIDTH * DIM, 7.2),                            // f_b
            synth(DIM * HIDDEN, 8.3),                           // g_a
            synth(WIDTH * DIM, 9.4),                            // g_b
            synth(HEADS * HIDDEN, 10.5),                        // b_proj
            synth(HEADS, 11.6),                                 // a_log
            synth(WIDTH, 12.7),                                 // dt_bias
            synth(DIM, 13.8).iter().map(|v| v + 1.0).collect(), // o_norm
        ],
    }
}

impl Fixture {
    fn kda(&self) -> KdaDeviceWeights<'_> {
        let f = &self.kda_f32;
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_proj,
            projection_encoding: ExpertEncoding::Bf16,
            q_conv1d: &f[0],
            k_conv1d: &f[1],
            v_conv1d: &f[2],
            f_a_proj: &f[3],
            f_b_proj: &f[4],
            g_a_proj: &f[5],
            g_b_proj: &f[6],
            b_proj: &f[7],
            a_log: &f[8],
            dt_bias: &f[9],
            o_norm: &f[10],
            norm_eps: EPS,
        }
    }

    fn layer<'a>(&'a self, state: &'a KdaDeviceState) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: AttentionSpec::Kda {
                weights: self.kda(),
                shape: shape(),
                state,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &self.router_weight,
                router_bias: &self.router_bias,
                gate: projection(&self.bank_gate, &self.residency, &self.shared_gate),
                up: projection(&self.bank_up, &self.residency, &self.shared_up),
                down: projection(&self.bank_down, &self.residency, &self.shared_down),
                inter: INTER,
                top_k: TOP_K,
                renormalize: true,
                branch_scale: BRANCH_SCALE,
            }),
            norm_eps: EPS,
        }
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (ms + eps).sqrt().recip();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// The host reference for the router: sigmoid, correction bias, top-k
/// with ties to the lower index, renormalise, scale.
fn route(x: &[f32], w: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let logits: Vec<f32> = (0..EXPERTS)
        .map(|e| {
            w[e * HIDDEN..(e + 1) * HIDDEN]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let scores: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let sel: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut ranked: Vec<usize> = (0..EXPERTS).collect();
    ranked.sort_by(|&a, &b| {
        sel[b]
            .partial_cmp(&sel[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let ids: Vec<usize> = ranked[..TOP_K].to_vec();
    let gathered: Vec<f32> = ids.iter().map(|&i| scores[i]).collect();
    let sum: f32 = gathered.iter().sum::<f32>() + 1e-20;
    let weights = gathered.iter().map(|w| w / sum * BRANCH_SCALE).collect();
    (ids, weights)
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

/// One projection's bank reference — a free function so both the KDA
/// and MLA layer gates can build one without fighting the borrow
/// checker over a closure's return lifetime.
fn bank<'a>(w: &'a [u8], offsets: &'a [ExpertOffset]) -> ExpertBankRef<'a> {
    ExpertBankRef {
        weights: w,
        offsets,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The whole layer, against a reference that composes the already-gated
/// attention and expert paths with a host router and combine.
#[test]
fn the_layer_matches_a_reference_composed_of_its_gated_parts() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let got = m
        .kimi_decoder_layer_traced(f.layer(&state), &f.x)
        .expect("device layer");

    // The reference, stage by stage, on a state that advances the same way.
    let ref_state = KdaDeviceState::zeros(&m, shape());
    let normed = rms_norm(&f.x, &f.input_norm, EPS);
    let (attn, _) = m
        .kda_attention_step(f.kda(), shape(), &ref_state, &normed)
        .expect("attention");
    let after: Vec<f32> = f.x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let post = rms_norm(&after, &f.post_norm, EPS);
    let (ids, weights) = route(&post, &f.router_weight, &f.router_bias);

    assert_eq!(got.input_normed, normed, "input norm");
    assert!(max_abs(&got.attention, &attn) < TOLERANCE, "attention");
    assert!(
        max_abs(&got.after_attention, &after) < TOLERANCE,
        "residual"
    );
    assert!(
        max_abs(&got.post_attention_normed, &post) < TOLERANCE,
        "post norm"
    );
    assert_eq!(
        got.selected_ids,
        ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        "the device chose other experts"
    );
    assert!(
        max_abs(&got.combine_weights[..TOP_K], &weights) < TOLERANCE,
        "routed weights"
    );
    assert_eq!(
        got.combine_weights[TOP_K], 1.0,
        "the shared branch is unscaled"
    );
    assert_eq!(
        got.expert_offsets,
        ids.iter().map(|&i| f.residency[i]).collect::<Vec<_>>(),
        "the GPU-written ROUTED offset table"
    );

    // The routed experts, through the already-gated block path — routed
    // slots only, because the shared branch no longer lives in the
    // routed bank.
    let offsets: Vec<ExpertOffset> = ids.iter().map(|&i| ExpertOffset(f.residency[i])).collect();
    let (outs, _) = m
        .bf16_moe_ffn_blocks(
            &[MoeBlockCall {
                banks: MoeFfnBanks {
                    gate: bank(&f.bank_gate, &offsets),
                    up: bank(&f.bank_up, &offsets),
                    down: bank(&f.bank_down, &offsets),
                    hidden: HIDDEN,
                    inter: INTER,
                },
                x: &post,
            }],
            BlockLowering::Separate,
        )
        .expect("experts");
    // The shared branch: the same gated block path over its own
    // regions, one slot at offset zero.
    let zero = [ExpertOffset(0)];
    let (shared_outs, _) = m
        .bf16_moe_ffn_blocks(
            &[MoeBlockCall {
                banks: MoeFfnBanks {
                    gate: bank(&f.shared_gate, &zero),
                    up: bank(&f.shared_up, &zero),
                    down: bank(&f.shared_down, &zero),
                    hidden: HIDDEN,
                    inter: INTER,
                },
                x: &post,
            }],
            BlockLowering::Separate,
        )
        .expect("shared expert");
    let all_outs: Vec<f32> = outs[0].iter().chain(&shared_outs[0]).copied().collect();
    assert!(
        max_abs(&got.expert_outputs, &all_outs) < TOLERANCE,
        "per-slot expert outputs"
    );

    let mut want = after.clone();
    for (slot, w) in got.combine_weights.iter().enumerate() {
        for (j, o) in want.iter_mut().enumerate() {
            *o += w * all_outs[slot * HIDDEN + j];
        }
    }
    assert!(max_abs(&got.output, &want) < TOLERANCE, "layer output");
}

/// A route naming a non-resident expert is refused, not served.
#[test]
fn a_non_resident_selection_is_refused() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let mut evicted = f.residency.clone();
    // Evict every resident expert but one, so the route must leave the
    // bank.
    for slot in evicted.iter_mut().take(RESIDENT).skip(1) {
        *slot = layer_shader::NOT_RESIDENT;
    }
    let mut w = f.layer(&state);
    {
        let m = moe_mut(&mut w);
        let a = ExpertAddressing::Table(&evicted);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    assert!(matches!(
        m.kimi_decoder_layer(w, &f.x),
        Err(GroupedError::LayerRouteNotResident { layer: 0, .. })
    ));
    // Still usable — a refusal must not have left the backend wedged.
    assert!(m.kimi_decoder_layer(f.layer(&state), &f.x).is_ok());
}

/// Host-side shape faults refuse before anything is encoded.
#[test]
fn shape_faults_are_refused() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());

    let mut no_experts = f.layer(&state);
    moe_mut(&mut no_experts).top_k = 0;
    assert_eq!(
        m.kimi_decoder_layer(no_experts, &f.x).map(|(o, _)| o),
        Err(GroupedError::NoExpertsSelected)
    );

    let short = vec![0u32; EXPERTS - 1];
    let mut bad_residency = f.layer(&state);
    {
        let m = moe_mut(&mut bad_residency);
        let a = ExpertAddressing::Table(&short);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    assert!(matches!(
        m.kimi_decoder_layer(bad_residency, &f.x),
        Err(GroupedError::SlotCountMismatch { .. })
    ));

    let mut truncated = f.layer(&state);
    let half = &f.bank_down[..f.bank_down.len() / 2];
    moe_mut(&mut truncated).down.routed.bytes = half;
    assert!(matches!(
        m.kimi_decoder_layer(truncated, &f.x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let mut too_many = f.layer(&state);
    let wide = vec![0u32; layer_shader::MAX_EXPERTS + 1];
    let wide_w = vec![0.0f32; (layer_shader::MAX_EXPERTS + 1) * HIDDEN];
    let wide_b = vec![0.0f32; layer_shader::MAX_EXPERTS + 1];
    {
        let m = moe_mut(&mut too_many);
        let a = ExpertAddressing::Table(&wide);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    moe_mut(&mut too_many).router_weight = &wide_w;
    moe_mut(&mut too_many).router_bias = &wide_b;
    assert!(matches!(
        m.kimi_decoder_layer(too_many, &f.x),
        Err(GroupedError::SlotCountMismatch { .. })
    ));
}

/// The plain entry point agrees with the traced one, and the traced
/// planes are the lengths their names imply.
#[test]
fn the_traced_layer_agrees_with_the_plain_one() {
    let m = backend();
    let f = fixture();
    let a = KdaDeviceState::zeros(&m, shape());
    let (plain, gpu) = m.kimi_decoder_layer(f.layer(&a), &f.x).unwrap();
    let b = KdaDeviceState::zeros(&m, shape());
    let traced = m.kimi_decoder_layer_traced(f.layer(&b), &f.x).unwrap();

    assert_eq!(traced.output, plain, "tracing must not change the answer");
    assert!(gpu >= 0.0 && traced.gpu_ms >= 0.0);
    assert_eq!(traced.router_logits.len(), EXPERTS);
    assert_eq!(traced.router_scores.len(), EXPERTS);
    assert_eq!(traced.router_selection_scores.len(), EXPERTS);
    assert_eq!(traced.selected_ids.len(), TOP_K);
    assert_eq!(traced.combine_weights.len(), TOP_K + 1);
    // Routed offsets only: the shared branch resolves no address.
    assert_eq!(traced.expert_offsets.len(), TOP_K);
    assert_eq!(traced.expert_outputs.len(), (TOP_K + 1) * HIDDEN);
    assert_eq!(traced.input_normed.len(), HIDDEN);
    assert_eq!(traced.attention.len(), HIDDEN);
}

// ── R6b: the same decoder layer with MLA attention ──────────────────

const LATENT: usize = 8;
const NOPE: usize = 4;
const ROPE: usize = 2;
const V_DIM: usize = 4;

fn mla_shape() -> MlaShape {
    MlaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        kv_lora_rank: LATENT,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE,
        v_head_dim: V_DIM,
    }
}

/// MLA's four wide matrices as bf16 codes, plus its latent norm.
struct MlaBits {
    q: Vec<u8>,
    ka: Vec<u8>,
    kb: Vec<u8>,
    o: Vec<u8>,
    norm: Vec<f32>,
}

fn mla_bits() -> MlaBits {
    let s = mla_shape();
    MlaBits {
        q: bf16_bytes(HEADS * s.q_head_dim(), HIDDEN, 31.0),
        ka: bf16_bytes(s.cache_stride(), HIDDEN, 32.0),
        kb: bf16_bytes(s.kv_row(), LATENT, 33.0),
        o: bf16_bytes(HIDDEN, s.value_width(), 34.0),
        norm: synth(LATENT, 35.0).iter().map(|v| v + 1.0).collect(),
    }
}

impl MlaBits {
    fn device(&self) -> MlaDeviceWeights<'_> {
        MlaDeviceWeights {
            q_proj: &self.q,
            kv_a_proj: &self.ka,
            kv_a_norm: &self.norm,
            kv_b_proj: &self.kb,
            o_proj: &self.o,
            kv_a_norm_eps: EPS,
            projection_encoding: ExpertEncoding::Bf16,
        }
    }
}

/// **R6b.** The decoder layer with MLA attention instead of KDA, against
/// a reference composed of its already-gated parts.
///
/// The layer path is written once and takes the attention as a
/// parameter, so this checks that the MLA arm binds the same way — and
/// that the cache advances exactly once per layer call, which is the one
/// thing a chained encode can get wrong (the latent is only really
/// cached once the dispatch that wrote it has run).
#[test]
fn an_mla_decoder_layer_matches_a_reference_composed_of_its_parts() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);

    fn mla_layer<'a>(
        f: &'a Fixture,
        bits: &'a MlaBits,
        st: &'a MlaDeviceState,
    ) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &f.input_norm,
            post_attention_norm: &f.post_norm,
            attention: AttentionSpec::Mla {
                weights: bits.device(),
                shape: mla_shape(),
                state: st,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &f.router_weight,
                router_bias: &f.router_bias,
                gate: projection(&f.bank_gate, &f.residency, &f.shared_gate),
                up: projection(&f.bank_up, &f.residency, &f.shared_up),
                down: projection(&f.bank_down, &f.residency, &f.shared_down),
                inter: INTER,
                top_k: TOP_K,
                renormalize: true,
                branch_scale: BRANCH_SCALE,
            }),
            norm_eps: EPS,
        }
    }

    // Two positions, so the cache is genuinely read on the second.
    let ref_state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);
    for pos in 0..2 {
        let x: Vec<f32> = f.x.iter().map(|v| v + pos as f32 * 0.13).collect();
        let got = m
            .kimi_decoder_layer_traced(mla_layer(&f, &bits, &state), &x)
            .expect("mla decoder layer");
        assert_eq!(
            state.len(),
            pos + 1,
            "the MLA cache must advance exactly once a layer call"
        );

        // The reference: the gated attention, then the host's own
        // residual / norm / router / combine.
        let normed = rms_norm(&x, &f.input_norm, EPS);
        let (attn, _) = m
            .mla_attention_step(bits.device(), mla_shape(), &ref_state, &normed)
            .expect("attention alone");
        let after: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let post = rms_norm(&after, &f.post_norm, EPS);
        let (ids, weights) = route(&post, &f.router_weight, &f.router_bias);

        assert!(
            max_abs(&got.attention, &attn) < TOLERANCE,
            "pos {pos} attention"
        );
        assert!(
            max_abs(&got.after_attention, &after) < TOLERANCE,
            "pos {pos} residual"
        );
        assert!(
            max_abs(&got.post_attention_normed, &post) < TOLERANCE,
            "pos {pos} post norm"
        );
        assert_eq!(
            got.selected_ids,
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            "pos {pos}: the device chose other experts"
        );
        assert!(
            max_abs(&got.combine_weights[..TOP_K], &weights) < TOLERANCE,
            "pos {pos} routed weights"
        );

        let offsets: Vec<ExpertOffset> =
            ids.iter().map(|&i| ExpertOffset(f.residency[i])).collect();
        let (outs, _) = m
            .bf16_moe_ffn_blocks(
                &[MoeBlockCall {
                    banks: MoeFfnBanks {
                        gate: bank(&f.bank_gate, &offsets),
                        up: bank(&f.bank_up, &offsets),
                        down: bank(&f.bank_down, &offsets),
                        hidden: HIDDEN,
                        inter: INTER,
                    },
                    x: &post,
                }],
                BlockLowering::Separate,
            )
            .expect("experts");
        let zero = [ExpertOffset(0)];
        let (shared_outs, _) = m
            .bf16_moe_ffn_blocks(
                &[MoeBlockCall {
                    banks: MoeFfnBanks {
                        gate: bank(&f.shared_gate, &zero),
                        up: bank(&f.shared_up, &zero),
                        down: bank(&f.shared_down, &zero),
                        hidden: HIDDEN,
                        inter: INTER,
                    },
                    x: &post,
                }],
                BlockLowering::Separate,
            )
            .expect("shared expert");
        let all_outs: Vec<f32> = outs[0].iter().chain(&shared_outs[0]).copied().collect();
        let mut want = after.clone();
        for (slot, w) in got.combine_weights.iter().enumerate() {
            for (j, o) in want.iter_mut().enumerate() {
                *o += w * all_outs[slot * HIDDEN + j];
            }
        }
        assert!(
            max_abs(&got.output, &want) < TOLERANCE,
            "pos {pos} layer output"
        );
    }
}

/// A KDA layer and an MLA layer in ONE command buffer — the shape R6c
/// runs at real weights. Both caches must advance, and the second layer
/// must read the first's output.
#[test]
fn a_kda_layer_and_an_mla_layer_share_one_command_buffer() {
    let m = backend();
    let f = fixture();
    let bits = mla_bits();
    let kda_state = KdaDeviceState::zeros(&m, shape());
    let mla_state = MlaDeviceState::with_capacity(&m, mla_shape(), 8);

    let mut second = f.layer(&kda_state);
    second.attention = AttentionSpec::Mla {
        weights: bits.device(),
        shape: mla_shape(),
        state: &mla_state,
    };
    let chain = [
        KimiLayerCall {
            weights: f.layer(&kda_state),
        },
        KimiLayerCall { weights: second },
    ];
    let planes = m
        .kimi_decoder_layers_traced(&chain, &f.x)
        .expect("mixed chain");
    assert_eq!(planes.len(), 2);
    assert_eq!(mla_state.len(), 1, "the MLA cache advanced once");

    // Layer 1 read layer 0's OUTPUT, not the original input: its input
    // norm is the norm OF that output, which a layer fed the original
    // `x` could not produce.
    let want = rms_norm(&planes[0].output, &f.input_norm, EPS);
    assert!(
        max_abs(&planes[1].input_normed, &want) < TOLERANCE,
        "layer 1's input norm is not the norm of layer 0's output"
    );
    let wrong = rms_norm(&f.x, &f.input_norm, EPS);
    assert!(
        max_abs(&planes[1].input_normed, &wrong) > TOLERANCE,
        "control: the two candidate inputs must be distinguishable"
    );
}

mod addressing;
mod dense;
mod encoding;
mod head;
mod shared;

/// **The route trace reports what the router actually decided.**
///
/// Checked against the host reference rather than against itself, and
/// against the traced path's own reading of the same buffer — so the
/// cheap instrumentation is pinned to the expensive one that is already
/// gated, not merely to itself.
#[test]
fn the_route_trace_matches_the_routers_own_decision() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let calls = [KimiLayerCall {
        weights: f.layer(&state),
    }];

    let mut trace = ExecutionTrace::default();
    let (out, _) = b
        .kimi_decoder_layers(&calls, &f.x, Some(&mut trace))
        .expect("chain runs");
    assert_eq!(trace.routes.len(), 1, "one entry a layer");
    assert_eq!(trace.routes[0].len(), TOP_K);

    // The host reference for the same input.
    let normed = rms_norm(&f.x, &f.input_norm, EPS);
    let attention = {
        let s2 = KdaDeviceState::zeros(&b, shape());
        let p = b
            .kimi_decoder_layer_traced(f.layer(&s2), &f.x)
            .expect("traced");
        p.attention
    };
    let after: Vec<f32> = f.x.iter().zip(&attention).map(|(a, c)| a + c).collect();
    let post = rms_norm(&after, &f.post_norm, EPS);
    let (want_ids, _) = route(&post, &f.router_weight, &f.router_bias);
    let want: Vec<u32> = want_ids.iter().map(|i| *i as u32).collect();
    assert_eq!(
        trace.routes[0], want,
        "the trace must carry the router's OWN selection, in router order"
    );
    let _ = normed;

    // Serving passes `None`, and must get the same answer — the
    // instrumentation may not perturb what it observes.
    let state_b = KdaDeviceState::zeros(&b, shape());
    let (untraced, _) = b
        .kimi_decoder_layers(
            &[KimiLayerCall {
                weights: f.layer(&state_b),
            }],
            &f.x,
            None,
        )
        .expect("untraced");
    assert_eq!(untraced, out, "tracing must not change the answer");
}

/// **Identity addressing is the same arithmetic, not a second path.**
///
/// A full bank whose experts sit at their own index must give exactly
/// what a table spelling out those offsets gives — so `Identity` is a
/// way of NOT tabulating an address, never a different lowering.
///
/// The fixture's bank is packed, so the comparison is made on the one
/// shape where both can describe the same thing: a table that happens to
/// be the identity map.
#[test]
fn identity_addressing_equals_a_table_that_spells_out_the_same_offsets() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let stride = (INTER * HIDDEN * 2) as u32;
    // A bank holding every scored expert at its own index.
    let per = INTER * HIDDEN * 2;
    let full_gate: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(INTER, HIDDEN, 3.0 + e as f32))
        .collect();
    let full_up: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(INTER, HIDDEN, 9.0 + e as f32))
        .collect();
    let full_down: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| bf16_bytes(HIDDEN, INTER, 15.0 + e as f32))
        .collect();
    assert_eq!(full_gate.len(), EXPERTS * per);
    let spelled: Vec<u32> = (0..EXPERTS).map(|e| (e * per) as u32).collect();

    let run = |addressing: ExpertAddressing<'_>| {
        let state = KdaDeviceState::zeros(&b, shape());
        let mut w = f.layer(&state);
        // The shared branch is its own region in both arms — identical,
        // so any difference is routed addressing.
        fn region(bytes: &[u8]) -> EncodedRegion<'_> {
            EncodedRegion {
                bytes,
                encoding: ExpertEncoding::Bf16,
            }
        }
        let m = KimiMoeWeights {
            router_weight: &f.router_weight,
            router_bias: &f.router_bias,
            gate: ProjectionBank {
                routed: region(&full_gate),
                addressing,
                shared: Some(region(&f.shared_gate)),
            },
            up: ProjectionBank {
                routed: region(&full_up),
                addressing,
                shared: Some(region(&f.shared_up)),
            },
            down: ProjectionBank {
                routed: region(&full_down),
                addressing,
                shared: Some(region(&f.shared_down)),
            },
            inter: INTER,
            top_k: TOP_K,
            renormalize: true,
            branch_scale: BRANCH_SCALE,
        };
        w.ffn = FfnSpec::Moe(m);
        b.kimi_decoder_layer(w, &f.x).expect("runs").0
    };

    let identity = run(ExpertAddressing::Identity {
        experts: EXPERTS,
        stride,
    });
    let tabulated = run(ExpertAddressing::Table(&spelled));
    assert_eq!(
        identity, tabulated,
        "identity addressing must be the same arithmetic as a table saying the same thing"
    );

    // And a full bank cannot refuse: every scored expert is addressable,
    // so there is no NOT_RESIDENT to hit however the router routes.
    let a = ExpertAddressing::Identity {
        experts: EXPERTS,
        stride,
    };
    for e in 0..EXPERTS {
        assert_eq!(a.offset_of(e), Some((e * per) as u32));
    }
    assert_eq!(
        a.offset_of(EXPERTS),
        None,
        "outside the bank is still refused"
    );
}
