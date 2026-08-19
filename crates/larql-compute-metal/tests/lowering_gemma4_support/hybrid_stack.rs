//! The two-layer Gemma 4 hybrid stack fixture for
//! `test_lowering_gemma4_arms.rs`: per-layer geometry (two head widths,
//! two rope tables), host operands, the MXFP4 expert banks in registered
//! regions, the descriptor-path `MoeLayerWeights` view, the CPU
//! reference per layer, and device residency.

use super::*;
use larql_compute::attention::rope::rope_freq_plan_proportional;
use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::lowering::attention::{AttnShape, LoweredPosition};
use larql_compute_metal::moe_descriptor::MoeExpertDescriptorTable;
use larql_compute_metal::MoeScratch;
use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

pub const LAYERS: usize = 2;
pub const DENSE_INTER: usize = 64;
pub const NUM_EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const MOE_INTER: usize = 32;
pub const S_T: usize = 4;
pub const S_POS: usize = S_T - 1;
/// Layer 0: narrow heads, plain rope; layer 1: wider heads, HF
/// `proportional` partial rotary — two tables of different length in one
/// stack, so `LayerLowering.inv_freq` must be honoured per layer.
pub const L0_NUM_Q: usize = 4;
pub const L0_NUM_KV: usize = 2;
pub const L0_HEAD_DIM: usize = 8;
pub const L0_THETA: f64 = 10_000.0;
pub const L1_NUM_Q: usize = 2;
pub const L1_NUM_KV: usize = 1;
pub const L1_HEAD_DIM: usize = 16;
pub const L1_THETA: f64 = 1_000_000.0;
pub const L1_ROTARY_FRACTION: f64 = 0.25;
/// Both layers project to the same Q/KV widths, so one scratch serves.
pub const S_Q_ROWS: usize = L0_NUM_Q * L0_HEAD_DIM;
pub const S_KV_ROWS: usize = L0_NUM_KV * L0_HEAD_DIM;
const _: () = assert!(S_Q_ROWS == L1_NUM_Q * L1_HEAD_DIM);
const _: () = assert!(S_KV_ROWS == L1_NUM_KV * L1_HEAD_DIM);
/// Stack norms use the centred convention (`1 + w`, weights stored
/// centred) — a different plan fact from the raw-offset arms above.
pub const CENTRED_OFFSET: f32 = 1.0;
pub const CENTRED_WEIGHT_AMPLITUDE: f32 = 0.2;
/// The layer's post-FFN norm epsilon, distinct from the branch epsilon.
pub const POST_EPS: f32 = 1e-6;
pub const ROUTER_SCALE_AMPLITUDE: f32 = 0.6;
pub const PER_EXPERT_SCALE_AMPLITUDE: f32 = 0.8;
pub const LAYER_SCALES: [f32; LAYERS] = [0.75, 1.25];
/// Two composed hybrid layers over 4-bit experts, f32 both sides.
pub const STACK_PARITY: f64 = 1e-3;
/// Control factor on one selected expert's scale (a uniform factor would
/// vanish under the post-experts RMS norm).
pub const PER_EXPERT_PERTURB: f32 = 2.0;
pub const LAYER_SCALE_PERTURB: f32 = 1.0;
pub const SEED_STRIDE: u32 = 1_000;
pub const GU_ROWS: usize = 2 * MOE_INTER;
pub const GU_PAYLOAD_PER_EXPERT: usize = GU_ROWS * (HIDDEN / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
pub const GU_SCALES_PER_EXPERT: usize = GU_ROWS * (HIDDEN / MXFP4_GROUP_ELEMS);
pub const DN_PAYLOAD_PER_EXPERT: usize =
    HIDDEN * (MOE_INTER / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
pub const DN_SCALES_PER_EXPERT: usize = HIDDEN * (MOE_INTER / MXFP4_GROUP_ELEMS);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tweak {
    None,
    /// Multiply this expert's per-expert scale by `PER_EXPERT_PERTURB`.
    PerExpertScale(usize),
    /// Replace the layer scale by `LAYER_SCALE_PERTURB`.
    LayerScale,
}

pub struct LayerGeom {
    pub num_q: usize,
    pub num_kv: usize,
    pub head_dim: usize,
    pub theta: f64,
    pub inv_freq: Vec<f32>,
}

pub fn layer_geom(l: usize) -> LayerGeom {
    match l {
        0 => LayerGeom {
            num_q: L0_NUM_Q,
            num_kv: L0_NUM_KV,
            head_dim: L0_HEAD_DIM,
            theta: L0_THETA,
            inv_freq: (0..L0_HEAD_DIM / 2)
                .map(|i| L0_THETA.powf(-2.0 * i as f64 / L0_HEAD_DIM as f64) as f32)
                .collect(),
        },
        _ => LayerGeom {
            num_q: L1_NUM_Q,
            num_kv: L1_NUM_KV,
            head_dim: L1_HEAD_DIM,
            theta: L1_THETA,
            inv_freq: rope_freq_plan_proportional(L1_HEAD_DIM, L1_ROTARY_FRACTION, L1_THETA)
                .inv_freq_f32(),
        },
    }
}

pub struct StackLayer {
    pub geom: LayerGeom,
    pub attn_norm: Vec<f32>,
    pub q: (Vec<f32>, Vec<u8>),
    pub k: (Vec<f32>, Vec<u8>),
    pub v: (Vec<f32>, Vec<u8>),
    pub o: (Vec<f32>, Vec<u8>),
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub pre_ffn_norm: Vec<f32>,
    pub gate: (Vec<f32>, Vec<u8>),
    pub up: (Vec<f32>, Vec<u8>),
    pub down: (Vec<f32>, Vec<u8>),
    pub router_proj: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub router_conditioning: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
    pub gu_payload: AlignedRegion,
    pub gu_scales: AlignedRegion,
    pub dn_payload: AlignedRegion,
    pub dn_scales: AlignedRegion,
    pub expert_gate_up: Vec<Vec<f32>>,
    pub expert_down: Vec<Vec<f32>>,
    pub pre_experts_norm: Vec<f32>,
    pub post_dense_norm: Vec<f32>,
    pub post_experts_norm: Vec<f32>,
    pub post_ffn_norm: Vec<f32>,
    pub layer_scale: f32,
}

pub fn build_layer(l: usize, tweak: Tweak) -> StackLayer {
    let geom = layer_geom(l);
    let s = |i: u32| l as u32 * SEED_STRIDE + i;
    let mk16 = |n: usize, k: usize, seed: u32| f16_matrix(&det(n * k, seed, WEIGHT_AMPLITUDE));
    let centred = |seed: u32| det(HIDDEN, seed, CENTRED_WEIGHT_AMPLITUDE);
    let (q_rows, kv_rows) = (geom.num_q * geom.head_dim, geom.num_kv * geom.head_dim);

    let mut gu_payload = Vec::new();
    let mut gu_scales = Vec::new();
    let mut dn_payload = Vec::new();
    let mut dn_scales = Vec::new();
    let mut expert_gate_up = Vec::new();
    let mut expert_down = Vec::new();
    for e in 0..NUM_EXPERTS as u32 {
        let gu = Mxfp4Matrix::quantize(
            &det(GU_ROWS * HIDDEN, s(40 + 2 * e), WEIGHT_AMPLITUDE),
            GU_ROWS,
            HIDDEN,
        );
        let dn = Mxfp4Matrix::quantize(
            &det(HIDDEN * MOE_INTER, s(41 + 2 * e), WEIGHT_AMPLITUDE),
            HIDDEN,
            MOE_INTER,
        );
        expert_gate_up.push(gu.dequantized());
        expert_down.push(dn.dequantized());
        gu_payload.extend_from_slice(&gu.packed);
        gu_scales.extend_from_slice(&gu.scales);
        dn_payload.extend_from_slice(&dn.packed);
        dn_scales.extend_from_slice(&dn.scales);
    }
    assert_eq!(gu_payload.len(), NUM_EXPERTS * GU_PAYLOAD_PER_EXPERT);
    assert_eq!(dn_payload.len(), NUM_EXPERTS * DN_PAYLOAD_PER_EXPERT);

    let router_scale = near_one(HIDDEN, s(30), ROUTER_SCALE_AMPLITUDE);
    let root_inv = (HIDDEN as f32).powf(-0.5);
    let mut per_expert_scale = near_one(NUM_EXPERTS, s(31), PER_EXPERT_SCALE_AMPLITUDE);
    if let Tweak::PerExpertScale(e) = tweak {
        per_expert_scale[e] *= PER_EXPERT_PERTURB;
    }
    StackLayer {
        attn_norm: centred(s(1)),
        q: mk16(q_rows, HIDDEN, s(2)),
        k: mk16(kv_rows, HIDDEN, s(3)),
        v: mk16(kv_rows, HIDDEN, s(4)),
        o: mk16(HIDDEN, q_rows, s(5)),
        q_norm: near_one(geom.head_dim, s(6), NORM_WEIGHT_AMPLITUDE),
        k_norm: near_one(geom.head_dim, s(7), NORM_WEIGHT_AMPLITUDE),
        k_cache: det(S_T * kv_rows, s(8), HIDDEN_AMPLITUDE),
        v_cache: det(S_T * kv_rows, s(9), HIDDEN_AMPLITUDE),
        pre_ffn_norm: centred(s(10)),
        gate: mk16(DENSE_INTER, HIDDEN, s(11)),
        up: mk16(DENSE_INTER, HIDDEN, s(12)),
        down: mk16(HIDDEN, DENSE_INTER, s(13)),
        router_proj: det(NUM_EXPERTS * HIDDEN, s(14), WEIGHT_AMPLITUDE),
        router_conditioning: router_scale.iter().map(|v| v * root_inv).collect(),
        router_scale,
        per_expert_scale,
        gu_payload: AlignedRegion::from_bytes(&gu_payload),
        gu_scales: AlignedRegion::from_bytes(&gu_scales),
        dn_payload: AlignedRegion::from_bytes(&dn_payload),
        dn_scales: AlignedRegion::from_bytes(&dn_scales),
        expert_gate_up,
        expert_down,
        pre_experts_norm: centred(s(15)),
        post_dense_norm: centred(s(16)),
        post_experts_norm: centred(s(17)),
        post_ffn_norm: centred(s(18)),
        layer_scale: if tweak == Tweak::LayerScale {
            LAYER_SCALE_PERTURB
        } else {
            LAYER_SCALES[l]
        },
        geom,
    }
}

pub fn expert_slices(bank: &[u8], per: usize) -> Vec<&[u8]> {
    (0..NUM_EXPERTS)
        .map(|e| &bank[e * per..(e + 1) * per])
        .collect()
}

impl StackLayer {
    /// The routed bank as the descriptor MoE path consumes it. The
    /// routing policy here only has to satisfy the descriptor path's
    /// support check — the hybrid lowering performs its own conditioning,
    /// selection, renormalisation and per-expert scaling.
    pub fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            experts_gate_up: expert_slices(self.gu_payload.as_slice(), GU_PAYLOAD_PER_EXPERT),
            experts_down: expert_slices(self.dn_payload.as_slice(), DN_PAYLOAD_PER_EXPERT),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(self.gu_scales.as_slice(), GU_SCALES_PER_EXPERT),
                down: expert_slices(self.dn_scales.as_slice(), DN_SCALES_PER_EXPERT),
            },
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            expert_data_format: QuantFormat::MXFP4,
            router_proj: &self.router_proj,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &self.pre_experts_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: MOE_INTER,
            gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
        }
    }

    pub fn hybrid_ref(&self) -> HybridRef<'_> {
        HybridRef {
            hidden: HIDDEN,
            dense_inter: DENSE_INTER,
            moe_inter: MOE_INTER,
            top_k: TOP_K,
            pre_ffn_norm: &self.pre_ffn_norm,
            dense_gate: &self.gate.0,
            dense_up: &self.up.0,
            dense_down: &self.down.0,
            router_proj: &self.router_proj,
            router_scale: &self.router_scale,
            per_expert_scale: &self.per_expert_scale,
            expert_gate_up: &self.expert_gate_up,
            expert_down: &self.expert_down,
            layout: MoeFusedRowLayout::ContiguousHalves,
            pre_experts_norm: &self.pre_experts_norm,
            post_dense_norm: &self.post_dense_norm,
            post_experts_norm: &self.post_experts_norm,
            post_ffn_norm: &self.post_ffn_norm,
            eps: EPS,
            post_eps: POST_EPS,
            offset: CENTRED_OFFSET,
            layer_scale: self.layer_scale,
        }
    }

    /// CPU attention for this layer: weighted QK norm + V norm + rope
    /// from ITS table, no sinks/gate/post-norm.
    pub fn cpu_attention(&self, h: &[f32]) -> Vec<f32> {
        cpu_attention(
            h,
            &AttnGeometry {
                hidden: HIDDEN,
                num_q: self.geom.num_q,
                num_kv: self.geom.num_kv,
                head_dim: self.geom.head_dim,
                t: S_T,
                pos: S_POS,
                eps: EPS,
                norm_offset: CENTRED_OFFSET,
                qk_eps: EPS,
                qk_offset: RAW_OFFSET,
                score_scale: 1.0 / (self.geom.head_dim as f32).sqrt(),
                inv_freq: &self.geom.inv_freq,
            },
            &AttnOperands {
                norm_w: &self.attn_norm,
                wq: &self.q.0,
                wk: &self.k.0,
                wv: &self.v.0,
                wo: &self.o.0,
                k_cache: &self.k_cache,
                v_cache: &self.v_cache,
                q_norm: Some(&self.q_norm),
                k_norm: Some(&self.k_norm),
                v_norm: true,
                v_source: VSource::Projection,
            },
        )
    }

    pub fn attn_shape(&self) -> AttnShape {
        AttnShape {
            hidden: HIDDEN,
            num_q_heads: self.geom.num_q,
            num_kv_heads: self.geom.num_kv,
            head_dim: self.geom.head_dim,
            norm_eps: EPS,
            norm_weight_offset: CENTRED_OFFSET,
            qk_norm_eps: EPS,
            parameter_free_q: false,
            parameter_free_k: false,
            parameter_free_v: true,
            query_scale: None,
            score_scale: 1.0 / (self.geom.head_dim as f32).sqrt(),
            position: LoweredPosition::Rope {
                theta: self.geom.theta,
            },
            window: None,
            softcap: None,
            position_index: S_POS,
            kv_len: S_T,
        }
    }
}

/// CPU: attention then the hybrid FFN per layer, capturing every layer.
pub fn cpu_stack(h0: &[f32], layers: &[StackLayer]) -> Vec<Vec<f32>> {
    let mut h = h0.to_vec();
    layers
        .iter()
        .map(|l| {
            h = hybrid_ffn_reference(&l.cpu_attention(&h), &l.hybrid_ref());
            h.clone()
        })
        .collect()
}

/// Device-side residency for one layer, alive for the whole stack.
pub struct LayerDevice {
    pub f16: [metal::Buffer; 7],
    pub f32: Vec<metal::Buffer>,
    pub scratch: MoeScratch,
    pub table: std::sync::Arc<MoeExpertDescriptorTable>,
}

/// Indices into `LayerDevice::f32`.
pub const ATTN_NORM: usize = 0;
pub const Q_NORM: usize = 1;
pub const K_NORM: usize = 2;
pub const K_CACHE: usize = 3;
pub const V_CACHE: usize = 4;
pub const INV_FREQ: usize = 5;
pub const PRE_FFN: usize = 6;
pub const ROUTER_COND: usize = 7;
pub const PE_SCALE: usize = 8;
pub const PRE_EXPERTS: usize = 9;
pub const POST_DENSE: usize = 10;
pub const POST_EXPERTS: usize = 11;
pub const POST_FFN: usize = 12;

pub fn upload_layer(gpu: &MetalBackend, idx: usize, l: &StackLayer) -> LayerDevice {
    for bank in [&l.gu_payload, &l.gu_scales, &l.dn_payload, &l.dn_scales] {
        assert!(gpu.lowering_register_region(bank.as_slice()));
    }
    let moe = l.moe();
    let scratch = MoeScratch::new_public_with_format(
        gpu,
        TOP_K,
        HIDDEN,
        MOE_INTER,
        QuantFormat::MXFP4,
        HIDDEN,
    );
    assert!(
        gpu.lowering_moe_supported(&moe, &scratch),
        "layer {idx}: descriptor MoE path refused the hybrid's routed bank"
    );
    let table = gpu
        .lowering_moe_descriptor(idx, &moe, MOE_INTER, HIDDEN)
        .expect("expert slices lie in registered regions");
    let up = |v: &[f32]| gpu.lowering_upload(v).unwrap();
    LayerDevice {
        f16: [
            &l.q.1, &l.k.1, &l.v.1, &l.o.1, &l.gate.1, &l.up.1, &l.down.1,
        ]
        .map(|b| gpu.lowering_weight(b)),
        f32: vec![
            up(&l.attn_norm),
            up(&l.q_norm),
            up(&l.k_norm),
            up(&l.k_cache),
            up(&l.v_cache),
            up(&l.geom.inv_freq),
            up(&l.pre_ffn_norm),
            up(&l.router_conditioning),
            up(&l.per_expert_scale),
            up(&l.pre_experts_norm),
            up(&l.post_dense_norm),
            up(&l.post_experts_norm),
            up(&l.post_ffn_norm),
        ],
        scratch,
        table,
    }
}
