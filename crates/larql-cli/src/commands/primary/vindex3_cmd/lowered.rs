//! G6d: execute a `ComponentOpPlan` through the Metal lowering.
//!
//! Everything before this rung compared the lowering against a reference
//! transcribed from the plan, which establishes **plan → lowering**
//! fidelity and nothing more. It cannot catch the plan and the lowering
//! sharing a mistake — which is exactly what the omitted four-norm
//! semantics were, and they survived every internally-consistent gate.
//!
//! So this path runs the *real container* and its logits are comparable
//! against the independent Glimmer oracle.
//!
//! Lives in the CLI because that is where the device is injected:
//! `larql-vindex` never links Metal, and `larql-compute-metal` never sees
//! a plan. This module is the only place both are in scope, which keeps
//! the lowering primitives free of plan types and the plan free of device
//! types.

use std::collections::HashMap;

use larql_compute::backend::MatMul;
use larql_compute_metal::lowering::attention::{AttnShape, AttnWeights, LoweredPosition};
use larql_compute_metal::lowering::ffn::{FfnShape, FfnWeights};
use larql_compute_metal::lowering::head::{HeadScratch, HeadShape, HeadWeights};
use larql_compute_metal::lowering::stack::{
    LayerFfnLowering, LayerLowering, RoutedFfnLowering, StackScratch,
};
use larql_compute_metal::lowering::{DeviceBuffer, LoweredMatrix, PostNorm};
use larql_compute_metal::MetalBackend;
use larql_models::config::PositionPolicy;
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::graph::policy::AttentionSpan;
use larql_vindex::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::{
    load_weight, AlignedBytes, LoadedWeight,
};
use larql_vindex::format::vindex3::opplan::{
    ComponentOpPlan, FfnOp, LayerPlan, NormOp, OperandRef,
};

/// One matrix operand, resident on the device.
struct DeviceMatrix {
    /// `scales` is unused for f16; the representation is what the plan's
    /// per-class policy asked for, not something inferred here.
    packed: DeviceBuffer,
    scales: DeviceBuffer,
    tensor_scale: f32,
    format: WeightFormat,
    rows: usize,
    cols: usize,
}

impl DeviceMatrix {
    fn as_lowered(&self) -> LoweredMatrix<'_> {
        match self.format {
            WeightFormat::F16 => LoweredMatrix::F16 {
                bytes: &self.packed,
            },
            WeightFormat::Mxfp4 => LoweredMatrix::Mxfp4 {
                packed: &self.packed,
                scales: &self.scales,
            },
            _ => LoweredMatrix::Nvfp4 {
                packed: &self.packed,
                scales: &self.scales,
                tensor_scale: self.tensor_scale,
            },
        }
    }
}

/// Per-layer resident state. No host-side KV: the caches are device
/// buffers that survive across positions, which is the whole point.
struct LayerResident {
    q: DeviceMatrix,
    k: DeviceMatrix,
    v: DeviceMatrix,
    o: DeviceMatrix,
    q_bias: Option<DeviceBuffer>,
    k_bias: Option<DeviceBuffer>,
    v_bias: Option<DeviceBuffer>,
    o_bias: Option<DeviceBuffer>,
    sinks: Option<DeviceBuffer>,
    gate: Option<DeviceMatrix>,
    ffn: FfnResident,
    pre_attn_norm: DeviceBuffer,
    post_attn_norm: Option<(DeviceBuffer, f32, f32)>,
    pre_ffn_norm: DeviceBuffer,
    post_ffn_norm: Option<(DeviceBuffer, f32, f32)>,
    k_cache: DeviceBuffer,
    v_cache: DeviceBuffer,
}

/// A layer's resident FFN: dense gate/up/down matrices, or a routed
/// expert bank resolved to registered regions plus the served MoE
/// scratch and descriptor table.
enum FfnResident {
    Dense {
        gate: DeviceMatrix,
        up: DeviceMatrix,
        down: DeviceMatrix,
    },
    Routed(Box<RoutedLayer>),
}

/// One routed layer's expert bank and MoE machinery, held for the
/// session: the packed expert bytes in page-aligned, region-registered
/// buffers (bound zero-copy, never copied per token), the f32 router and
/// bias operands, the per-layer `MoeScratch`, and the descriptor table.
struct RoutedLayer {
    gate_up_blocks: AlignedBytes,
    gate_up_scales: AlignedBytes,
    down_blocks: AlignedBytes,
    down_scales: AlignedBytes,
    router_proj: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    gu_expert_bytes: usize,
    gu_scale_bytes: usize,
    dn_expert_bytes: usize,
    dn_scale_bytes: usize,
    experts: usize,
    top_k: usize,
    inter: usize,
    gate_rule: larql_compute::MoeGateRule,
    table: std::sync::Arc<larql_compute_metal::moe_descriptor::MoeExpertDescriptorTable>,
    scratch: larql_compute_metal::MoeScratch,
    eps: f32,
}

/// Per-expert byte slices into a packed bank: expert `e` occupies
/// `[e*per .. (e+1)*per]` of the bank's logical bytes.
fn expert_slices(bank: &AlignedBytes, per: usize, experts: usize) -> Vec<&[u8]> {
    let all = &bank.as_slice()[..per * experts];
    (0..experts).map(|e| &all[e * per..(e + 1) * per]).collect()
}

impl RoutedLayer {
    /// The `MoeLayerWeights` view the served descriptor path consumes,
    /// assembled from a `RoutedFfnOp` — per-expert slices into the
    /// registered banks, router/bias from f32 storage, GPT-OSS routing
    /// and gate semantics from the plan. Rebuilt per step (borrows are
    /// cheap; no bytes move).
    fn moe(&self) -> larql_compute::MoeLayerWeights<'_> {
        use larql_compute::{
            MoeExpertScales, MoeFusedRowLayout, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
        };
        larql_compute::MoeLayerWeights {
            experts_gate_up: expert_slices(
                &self.gate_up_blocks,
                self.gu_expert_bytes,
                self.experts,
            ),
            experts_down: expert_slices(&self.down_blocks, self.dn_expert_bytes, self.experts),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(&self.gate_up_scales, self.gu_scale_bytes, self.experts),
                down: expert_slices(&self.down_scales, self.dn_scale_bytes, self.experts),
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
            pre_experts_norm: &self.pre_ffn_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: self.experts,
            top_k: self.top_k,
            intermediate_size: self.inter,
            gate_rule: self.gate_rule,
        }
    }
}

/// Stages this run omits, for marginal-cost profiling.
///
/// Every one of these is an operation the plan marks **optional**, so
/// omitting it exercises a path the lowering already supports rather
/// than a special diagnostic branch. The numbers are wrong by
/// construction; the *time difference* is the measurement.
///
/// Ablation is used because this hardware supports counter sampling only
/// at compute-pass boundaries (`AtDispatchBoundary` is false on M3), so
/// per-dispatch GPU timestamps are unavailable. Splitting stages into
/// separate encoders to get boundaries would change what can overlap;
/// ablation leaves the schedule of everything that remains intact.
#[derive(Clone, Copy, Default)]
pub struct Ablation {
    pub no_query_scale: bool,
    pub no_rope: bool,
    pub no_qk_norm: bool,
    pub no_gate: bool,
    pub no_post_norms: bool,
}

impl Ablation {
    fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok();
        Self {
            no_query_scale: on("LARQL_ABLATE_QUERY_SCALE"),
            no_rope: on("LARQL_ABLATE_ROPE"),
            no_qk_norm: on("LARQL_ABLATE_QK_NORM"),
            no_gate: on("LARQL_ABLATE_GATE"),
            no_post_norms: on("LARQL_ABLATE_POST_NORMS"),
        }
    }

    fn any(&self) -> bool {
        self.no_query_scale || self.no_rope || self.no_qk_norm || self.no_gate || self.no_post_norms
    }
}

/// A plan lowered onto the device, ready to step positions.
pub struct LoweredSession<'a> {
    gpu: &'a MetalBackend,
    plan: &'a ComponentOpPlan,
    hidden: usize,
    /// Embedding stays f32 on the host: it is a row lookup, not matrix
    /// traffic, and only one row per token crosses to the device.
    embed_table: Vec<f32>,
    layers: Vec<LayerResident>,
    final_norm: Option<(DeviceBuffer, f32, f32)>,
    head: Option<DeviceMatrix>,
    head_multiplier: Option<f32>,
    head_softcap: Option<f32>,
    vocab: usize,
    scratch: Vec<DeviceBuffer>,
    inv_freq: HashMap<u64, DeviceBuffer>,
    position: usize,
    /// Destination for resolved GPU timestamps, when profiling.
    ablate: Ablation,
}

/// Load one matrix operand as NVFP4 and hand it to the device.
///
/// The buffers are keyed on the `AlignedBytes` address, which lives for
/// the session, so `lowering_weight` caches them and the weight is
/// uploaded once rather than per position.
fn resident_matrix(
    gpu: &MetalBackend,
    store: &OperandStore,
    operand: &OperandRef,
    format: WeightFormat,
    keep: &mut Vec<LoadedWeight>,
) -> Result<DeviceMatrix, VindexError> {
    let rows = operand.shape.first().copied().unwrap_or(0);
    let cols = operand.shape.get(1).copied().unwrap_or(0);
    let loaded = load_weight(store, operand, format)?;
    let m = match &loaded {
        LoadedWeight::Nvfp4 {
            packed,
            scales,
            tensor_scale,
        } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            tensor_scale: *tensor_scale,
            format: WeightFormat::Nvfp4,
            rows,
            cols,
        },
        LoadedWeight::Mxfp4 { packed, scales } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            tensor_scale: 1.0,
            format: WeightFormat::Mxfp4,
            rows,
            cols,
        },
        LoadedWeight::F16(bytes) => DeviceMatrix {
            packed: gpu.lowering_weight(bytes.as_slice()),
            scales: gpu.lowering_weight(&[]),
            tensor_scale: 1.0,
            format: WeightFormat::F16,
            rows,
            cols,
        },
        _ => {
            return Err(VindexError::Parse(format!(
                "operand `{}`: unsupported lowering format {format:?}",
                operand.tensor
            )))
        }
    };
    // The device buffers alias these allocations, so the session owns
    // them for its lifetime.
    keep.push(loaded);
    Ok(m)
}

/// Upload an optional f32 vector operand (a bias or the sink logits) to
/// the device, or `None` when the plan carries none.
fn resident_vector(
    gpu: &MetalBackend,
    store: &OperandStore,
    operand: Option<&OperandRef>,
) -> Result<Option<DeviceBuffer>, VindexError> {
    match operand {
        Some(op) => {
            let v = store.load(op)?;
            let buf = gpu
                .lowering_upload(&v)
                .ok_or_else(|| VindexError::Parse("vector operand upload failed".into()))?;
            Ok(Some(buf))
        }
        None => Ok(None),
    }
}

/// The `inv_freq` map key for a rotary policy — distinct per (theta,
/// scaled-or-plain) so YaRN and plain rope at the same base never share a
/// table; `None` for NoPE.
fn rope_table_key(position: &PositionPolicy) -> Option<u64> {
    match position {
        PositionPolicy::Rope { theta } => Some(theta.to_bits()),
        // Fold the yarn block into the key so two different blocks (or a
        // block vs plain rope) at one theta get their own tables. The
        // block's f64 fields hash deterministically.
        PositionPolicy::Yarn { theta, scaling } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            theta.to_bits().hash(&mut h);
            scaling.factor.to_bits().hash(&mut h);
            scaling.beta_fast.to_bits().hash(&mut h);
            scaling.beta_slow.to_bits().hash(&mut h);
            scaling
                .original_max_position_embeddings
                .to_bits()
                .hash(&mut h);
            scaling.truncate.hash(&mut h);
            Some(h.finish() | 1)
        }
        PositionPolicy::None => None,
    }
}

/// The inverse-frequency table for a rotary policy, matching the
/// interpreter kernel exactly: plain `theta^(-2i/d)` for rope, the YaRN
/// ramp for a scaled layer.
fn rope_inv_freq_table(position: &PositionPolicy, head_dim: usize) -> Vec<f32> {
    match position {
        PositionPolicy::Rope { theta } => (0..head_dim / 2)
            .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64) as f32)
            .collect(),
        PositionPolicy::Yarn { theta, scaling } => {
            let (inv_freq, _amplitude) =
                larql_vindex::format::vindex3::opplan::exec::kernels::yarn_frequencies(
                    scaling, head_dim, *theta,
                );
            inv_freq.iter().map(|f| *f as f32).collect()
        }
        PositionPolicy::None => Vec::new(),
    }
}

fn resident_norm(
    gpu: &MetalBackend,
    store: &OperandStore,
    op: &NormOp,
) -> Result<(DeviceBuffer, f32, f32), VindexError> {
    let w = store.load(&op.weight)?;
    let buf = gpu
        .lowering_upload(&w)
        .ok_or_else(|| VindexError::Parse("norm weight upload failed".into()))?;
    Ok((buf, op.eps as f32, op.weight_offset))
}

impl<'a> LoweredSession<'a> {
    /// Load every operand the plan consumes, once, resident on the
    /// device.
    /// `formats` is the plan's per-class policy, applied here rather
    /// than assumed: attention, FFN and head may each be resident in a
    /// different representation and still execute under one schedule.
    pub fn new(
        gpu: &'a MetalBackend,
        plan: &'a ComponentOpPlan,
        store: &OperandStore,
        formats: WeightFormats,
        max_positions: usize,
        keep: &mut Vec<LoadedWeight>,
    ) -> Result<Self, VindexError> {
        // YaRN, sinks and Q/K/V/O biases are lowered (A-9.4): the
        // amplitude rides slot 6 of the rope kernel, the sinks slot 10/11
        // of the attention kernel, the biases the `bias_add` kernel after
        // each projection, and a routed FFN through the served descriptor
        // MoE path (build_routed). A dense clamped-GLU FFN: the
        // lowering encodes plain gated FFNs only, and running the clamped
        // policy as plain gating would be a different model (A-9.4).
        if let Some(l) = plan.layers.iter().find(|l| {
            l.ffn
                .dense()
                .is_some_and(|f| !matches!(f.gate_policy, larql_models::ExpertGatePolicy::Gated))
        }) {
            return Err(VindexError::Parse(format!(
                "layer {} carries {:?}, which the Metal lowering does not execute yet (A-9.4); \
                 refusing rather than lowering it as plain gating",
                l.layer,
                l.ffn.dense().map(|f| f.gate_policy)
            )));
        }
        let embedding = plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("plan carries no embedding op".into()))?;
        let embed_table = store.load(&embedding.table)?;
        let hidden = embed_table.len() / embedding.vocab_size;

        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let a = &layer.attention;
            let kv_rows = a.num_kv_heads * a.head_dim;
            let zeros = vec![0.0f32; max_positions * kv_rows];
            layers.push(LayerResident {
                q: resident_matrix(gpu, store, &a.q, formats.attention, keep)?,
                k: resident_matrix(gpu, store, &a.k, formats.attention, keep)?,
                v: resident_matrix(gpu, store, &a.v, formats.attention, keep)?,
                o: resident_matrix(gpu, store, &a.o, formats.attention, keep)?,
                q_bias: resident_vector(gpu, store, a.q_bias.as_ref())?,
                k_bias: resident_vector(gpu, store, a.k_bias.as_ref())?,
                v_bias: resident_vector(gpu, store, a.v_bias.as_ref())?,
                o_bias: resident_vector(gpu, store, a.o_bias.as_ref())?,
                sinks: resident_vector(gpu, store, a.sinks.as_ref().map(|s| &s.logits))?,
                gate: match &a.output_gate {
                    Some(g) => Some(resident_matrix(
                        gpu,
                        store,
                        &g.projection,
                        formats.attention,
                        keep,
                    )?),
                    None => None,
                },
                ffn: build_ffn(gpu, store, layer, formats, keep)?,
                pre_attn_norm: resident_norm(gpu, store, &layer.pre_attention_norm)?.0,
                post_attn_norm: match &layer.post_attention_norm {
                    Some(op) => Some(resident_norm(gpu, store, op)?),
                    None => None,
                },
                pre_ffn_norm: resident_norm(gpu, store, &layer.pre_ffn_norm)?.0,
                post_ffn_norm: match &layer.post_ffn_norm {
                    Some(op) => Some(resident_norm(gpu, store, op)?),
                    None => None,
                },
                k_cache: gpu
                    .lowering_upload(&zeros)
                    .ok_or_else(|| VindexError::Parse("KV cache allocation failed".into()))?,
                v_cache: gpu
                    .lowering_upload(&zeros)
                    .ok_or_else(|| VindexError::Parse("KV cache allocation failed".into()))?,
            });
        }

        let final_norm = match &plan.final_norm {
            Some(op) => Some(resident_norm(gpu, store, op)?),
            None => None,
        };
        let (head, vocab, head_multiplier, head_softcap) = match &plan.output {
            Some(out) => {
                let m = resident_matrix(gpu, store, &out.projection, formats.head, keep)?;
                let v = m.rows;
                (
                    Some(m),
                    v,
                    out.multiplier.map(|m| m as f32),
                    out.softcapping,
                )
            }
            None => (None, 0, None, None),
        };

        // Scratch sized from the widest layer, allocated once.
        let max_q = plan
            .layers
            .iter()
            .map(|l| l.attention.num_q_heads * l.attention.head_dim)
            .max()
            .unwrap_or(hidden);
        let max_inter = plan
            .layers
            .iter()
            .filter_map(|l| l.ffn.dense().map(|f| f.intermediate_size))
            .max()
            .unwrap_or(hidden);
        // Slots 16 and 17 are both vocabulary-sized: the head writes raw
        // logits into one and the scaled/softcapped result into the
        // other. Sizing 16 as `hidden` made the readback fail closed —
        // `try_read_buffer_f32` refuses a buffer shorter than the
        // requested length, which is why this surfaced as "no output
        // head" rather than as garbage logits.
        let sizes = [
            hidden,
            hidden,
            hidden,
            max_q,
            max_q,
            max_q,
            hidden,
            hidden,
            hidden,
            max_inter,
            max_inter,
            max_inter,
            max_q,
            hidden,
            hidden,
            hidden,
            vocab.max(1),
            vocab.max(1),
        ];
        let scratch = sizes.iter().map(|n| gpu.lowering_scratch(*n)).collect();

        // One inverse-frequency table per distinct rotary policy in the
        // plan — keyed on (theta, yarn-or-plain), so a YaRN layer's ramped
        // frequencies and a plain layer's `theta^(-2i/d)` never collide on
        // theta alone. The table matches the interpreter's exactly: plain
        // rope from `rope_rotate`, YaRN from `kernels::yarn_frequencies`.
        let mut inv_freq: HashMap<u64, DeviceBuffer> = HashMap::new();
        for layer in &plan.layers {
            let a = &layer.attention;
            let key = rope_table_key(&a.position);
            if let Some(key) = key {
                inv_freq.entry(key).or_insert_with(|| {
                    let table = rope_inv_freq_table(&a.position, a.head_dim);
                    gpu.lowering_upload(&table).expect("inv_freq upload")
                });
            }
        }

        // Residency bootstrap. Without it the driver's wired-page
        // collector un-wires weights that sit idle between submissions,
        // and a decode walking ~15 GB per token pays a re-wire on every
        // touch — measured at 10x on a large f16 working set. One command
        // buffer referencing everything re-wires it at memcpy speed, and
        // steps fast enough thereafter keep themselves wired.
        //
        // The slices are the same allocations `lowering_weight` cached on,
        // so this wires the buffers the stack will actually bind.
        let mut streams: Vec<&[u8]> = Vec::with_capacity(keep.len() * 2);
        for w in keep.iter() {
            match w {
                LoadedWeight::Nvfp4 { packed, scales, .. } => {
                    streams.push(packed.as_slice());
                    streams.push(scales.as_slice());
                }
                LoadedWeight::Mxfp4 { packed, scales } => {
                    streams.push(packed.as_slice());
                    streams.push(scales.as_slice());
                }
                LoadedWeight::F16(b) => streams.push(b.as_slice()),
                _ => {}
            }
        }
        let wiring = std::time::Instant::now();
        gpu.wire_resident(&streams);
        eprintln!(
            "wired {} weight streams in {:.1} s",
            streams.len(),
            wiring.elapsed().as_secs_f64()
        );

        if inv_freq.len() > 1 {
            return Err(VindexError::Parse(format!(
                "plan carries {} distinct rotary tables; the lowered stack binds one shared \
                 inv_freq per token and cannot yet select per layer",
                inv_freq.len()
            )));
        }
        Ok(Self {
            gpu,
            plan,
            hidden,
            embed_table,
            layers,
            final_norm,
            head,
            head_multiplier,
            head_softcap,
            vocab,
            scratch,
            inv_freq,
            position: 0,
            ablate: Ablation::from_env(),
        })
    }

    /// Step one token: embed on the host, then the entire stack and head
    /// in **one** command buffer with a single wait.
    /// Step one token: embed on the host, then the entire stack and
    /// head in one command buffer with a single wait.
    pub fn step(&mut self, token: u32) -> Result<Option<Vec<f32>>, VindexError> {
        let t = self.position;
        let row = &self.embed_table[token as usize * self.hidden..][..self.hidden];
        let embedding = self
            .plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("no embedding".into()))?;
        let mut h0 = row.to_vec();
        if let Some(scale) = embedding.scale {
            h0.iter_mut().for_each(|v| *v *= scale);
        }
        // The judged embedding norm: Muse-Glimmer RMS-normalises every
        // looked-up row **weightlessly**. Nothing in the checkpoint
        // records that it happens — there is no operand to classify, so
        // no closure or parity gate over the container can see it, and
        // omitting it produced entirely plausible logits with the wrong
        // argmax (368 against the oracle's 13796). It was caught here
        // only by comparing against the independent model oracle, which
        // is precisely why that anchor exists.
        if let Some(norm) = embedding.norm {
            let ms = h0.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h0.len() as f64;
            let inv = 1.0 / (ms + norm.eps).sqrt();
            h0.iter_mut().for_each(|v| *v = (*v as f64 * inv) as f32);
        }
        let h_in = self
            .gpu
            .lowering_upload(&h0)
            .ok_or_else(|| VindexError::Parse("hidden upload failed".into()))?;

        let s = &self.scratch;
        let scratch = StackScratch {
            h_a: &s[0],
            h_b: &s[1],
            attn_normed: &s[2],
            q: &s[3],
            gate: &s[4],
            concat: &s[5],
            gated: &s[12],
            attn_out: &s[6],
            attn_post: &s[7],
            ffn_normed: &s[8],
            ffn_gate: &s[9],
            ffn_up: &s[10],
            ffn_act: &s[11],
            ffn_down: &s[13],
            ffn_post: &s[14],
            // Every rotary layer in this plan shares one table (checked
            // in `new`); a plan with several would need per-layer
            // selection, which the stack encoder does not yet express.
            inv_freq: self.inv_freq.values().next().unwrap_or(&self.scratch[0]),
        };

        let layers: Vec<LayerLowering> = self
            .plan
            .layers
            .iter()
            .zip(&self.layers)
            .map(|(plan_layer, r)| self.layer_lowering(plan_layer, r, t))
            .collect();

        let cmd = self.gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let h_final = self.gpu.encode_stack(enc, &h_in, &layers, &scratch, &[]);

        let logits_buf = match (&self.final_norm, &self.head) {
            (Some((nw, eps, off)), Some(head)) => {
                let hs = HeadScratch {
                    normed: &s[15],
                    raw_logits: &s[17],
                };
                let hw = HeadWeights {
                    projection: head.as_lowered(),
                    norm_weight: nw,
                };
                let shape = HeadShape {
                    hidden: self.hidden,
                    vocab: self.vocab,
                    norm_eps: *eps,
                    norm_weight_offset: *off,
                    multiplier: self.head_multiplier,
                    softcap: self.head_softcap,
                };
                self.gpu.encode_head(enc, h_final, &s[16], &hw, &hs, &shape);
                Some(&s[16])
            }
            _ => None,
        };
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = logits_buf.and_then(|b| self.gpu.lowering_readback(b, self.vocab));
        self.gpu.recycle_lowering_scratch(h_in);
        self.position += 1;
        Ok(out)
    }

    fn layer_lowering<'b>(
        &'b self,
        plan_layer: &'b LayerPlan,
        r: &'b LayerResident,
        t: usize,
    ) -> LayerLowering<'b> {
        let a = &plan_layer.attention;
        let post = |slot: &'b Option<(DeviceBuffer, f32, f32)>, scratch: &'b DeviceBuffer| {
            slot.as_ref().map(|(w, eps, off)| PostNorm {
                weight: w,
                eps: *eps,
                weight_offset: *off,
                scratch,
            })
        };
        LayerLowering {
            attn: AttnWeights {
                q: r.q.as_lowered(),
                k: r.k.as_lowered(),
                v: r.v.as_lowered(),
                o: r.o.as_lowered(),
                gate: r
                    .gate
                    .as_ref()
                    .filter(|_| !self.ablate.no_gate)
                    .map(DeviceMatrix::as_lowered),
                q_bias: r.q_bias.as_ref(),
                k_bias: r.k_bias.as_ref(),
                v_bias: r.v_bias.as_ref(),
                o_bias: r.o_bias.as_ref(),
                sinks: r.sinks.as_ref(),
                norm_weight: &r.pre_attn_norm,
                post_norm: post(&r.post_attn_norm, &self.scratch[7])
                    .filter(|_| !self.ablate.no_post_norms),
            },
            attn_shape: AttnShape {
                hidden: self.hidden,
                num_q_heads: a.num_q_heads,
                num_kv_heads: a.num_kv_heads,
                head_dim: a.head_dim,
                norm_eps: plan_layer.pre_attention_norm.eps as f32,
                norm_weight_offset: plan_layer.pre_attention_norm.weight_offset,
                // The interpreter passes the pre-attention norm's epsilon
                // as the QK-norm epsilon; it is not a separate fact.
                qk_norm_eps: plan_layer.pre_attention_norm.eps as f32,
                parameter_free_q: a.parameter_free_qk_norm.q && !self.ablate.no_qk_norm,
                parameter_free_k: a.parameter_free_qk_norm.k && !self.ablate.no_qk_norm,
                query_scale: a
                    .query_scale
                    .map(|s| s as f32)
                    .filter(|_| !self.ablate.no_query_scale),
                score_scale: a.score_scale as f32,
                position: match a.position {
                    _ if self.ablate.no_rope => LoweredPosition::None,
                    PositionPolicy::Rope { theta } => LoweredPosition::Rope { theta },
                    // YaRN's ramped `inv_freq` rides the shared table
                    // (built for this layer's policy in `new`); the
                    // amplitude rides slot 6 of the rope kernel.
                    PositionPolicy::Yarn { theta, scaling } => {
                        let amplitude =
                            larql_vindex::format::vindex3::opplan::exec::kernels::yarn_frequencies(
                                &scaling, a.head_dim, theta,
                            )
                            .1;
                        LoweredPosition::Scaled { theta, amplitude }
                    }
                    PositionPolicy::None => LoweredPosition::None,
                },
                // A window applies only to a sliding span; a full layer
                // attends the whole prefix whatever the plan records.
                window: match a.span {
                    AttentionSpan::Sliding => a.window,
                    _ => None,
                },
                softcap: a.logit_softcapping,
                position_index: t,
                kv_len: t + 1,
            },
            ffn: match &r.ffn {
                FfnResident::Dense { gate, up, down } => LayerFfnLowering::Dense {
                    weights: FfnWeights {
                        gate: gate.as_lowered(),
                        up: up.as_lowered(),
                        down: down.as_lowered(),
                        norm_weight: &r.pre_ffn_norm,
                        post_norm: post(&r.post_ffn_norm, &self.scratch[14])
                            .filter(|_| !self.ablate.no_post_norms),
                    },
                    shape: FfnShape {
                        hidden: self.hidden,
                        intermediate: plan_layer
                            .ffn
                            .dense()
                            .map_or(self.hidden, |f| f.intermediate_size),
                        norm_eps: plan_layer.pre_ffn_norm.eps as f32,
                        norm_weight_offset: plan_layer.pre_ffn_norm.weight_offset,
                    },
                },
                FfnResident::Routed(routed) => {
                    LayerFfnLowering::Routed(Box::new(RoutedFfnLowering {
                        moe: routed.moe(),
                        scratch: &routed.scratch,
                        table: &routed.table,
                        eps: routed.eps,
                    }))
                }
            },
            k_cache: &r.k_cache,
            v_cache: &r.v_cache,
        }
    }

    /// Matrix geometry the loader saw, for diagnostics.
    pub fn head_geometry(&self) -> Option<(usize, usize)> {
        self.head.as_ref().map(|h| (h.rows, h.cols))
    }

    /// Whether any stage is being ablated.
    pub fn ablation_active(&self) -> bool {
        self.ablate.any()
    }

    /// Whether the plan carried a final norm, for diagnostics.
    pub fn has_final_norm(&self) -> bool {
        self.final_norm.is_some()
    }

    /// Distinct rope bases the plan declares.
    pub fn rope_bases(&self) -> usize {
        self.inv_freq.len()
    }
}

/// Build a layer's resident FFN: dense matrices, or the routed expert
/// bank loaded into registered regions with its MoE machinery.
fn build_ffn(
    gpu: &MetalBackend,
    store: &OperandStore,
    layer: &LayerPlan,
    formats: WeightFormats,
    keep: &mut Vec<LoadedWeight>,
) -> Result<FfnResident, VindexError> {
    if let Some(op) = layer.ffn.routed() {
        return Ok(FfnResident::Routed(Box::new(build_routed(
            gpu, store, layer, op,
        )?)));
    }
    let dense = dense_ffn(layer)?;
    Ok(FfnResident::Dense {
        gate: resident_matrix(
            gpu,
            store,
            dense
                .gate
                .as_ref()
                .ok_or_else(|| VindexError::Parse("lowering requires a gated FFN".into()))?,
            formats.ffn,
            keep,
        )?,
        up: resident_matrix(gpu, store, &dense.up, formats.ffn, keep)?,
        down: resident_matrix(gpu, store, &dense.down, formats.ffn, keep)?,
    })
}

/// Load a routed layer's expert bank into page-aligned, region-registered
/// buffers and build its MoE scratch + descriptor table. The expert bytes
/// are bound zero-copy through the same registered-region path the served
/// `--routed-from` run uses — never copied per token.
fn build_routed(
    gpu: &MetalBackend,
    store: &OperandStore,
    layer: &LayerPlan,
    op: &larql_vindex::format::vindex3::opplan::RoutedFfnOp,
) -> Result<RoutedLayer, VindexError> {
    use larql_vindex::format::vindex3::opplan::exec::weights::AlignedBytes;
    // Only the packed-MXFP4 gpt-oss bank lowers today (native experts,
    // split scales); any other routed shape refuses rather than guess.
    if op.expert_format != larql_models::ExpertFormat::PackedMxfp4 {
        return Err(VindexError::Parse(format!(
            "layer {}: the Metal lowering serves only PackedMxfp4 routed experts, not {:?}",
            layer.layer, op.expert_format
        )));
    }
    let hidden = op.router.shape.get(1).copied().unwrap_or(0);
    let experts = op.experts;
    let inter = op.expert_intermediate_size;

    // Packed blocks and scales into aligned, registered banks.
    let aligned = |operand: &larql_vindex::format::vindex3::opplan::OperandRef| -> Result<AlignedBytes, VindexError> {
        let raw = store.load_raw(operand)?;
        Ok(AlignedBytes::from_bytes(&raw.bytes))
    };
    let gate_up_blocks = aligned(&op.gate_up.weights)?;
    let gate_up_scales = aligned(op.gate_up.scales.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed gate_up carries no scales",
            layer.layer
        ))
    })?)?;
    let down_blocks = aligned(&op.down.weights)?;
    let down_scales = aligned(op.down.scales.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {}: routed down carries no scales",
            layer.layer
        ))
    })?)?;
    for (bank, what) in [
        (&gate_up_blocks, "gate_up blocks"),
        (&gate_up_scales, "gate_up scales"),
        (&down_blocks, "down blocks"),
        (&down_scales, "down scales"),
    ] {
        if !gpu.lowering_register_region(bank.as_slice()) {
            return Err(VindexError::Parse(format!(
                "layer {}: could not register the routed {what} region (not page-aligned)",
                layer.layer
            )));
        }
    }
    let gu_expert_bytes = gate_up_blocks.logical_len() / experts;
    let gu_scale_bytes = gate_up_scales.logical_len() / experts;
    let dn_expert_bytes = down_blocks.logical_len() / experts;
    let dn_scale_bytes = down_scales.logical_len() / experts;

    let f32_or_empty =
        |o: Option<&larql_vindex::format::vindex3::opplan::OperandRef>| -> Result<Vec<f32>, VindexError> {
            match o {
                Some(op) => store.load(op),
                None => Ok(Vec::new()),
            }
        };
    let router_proj = store.load(&op.router)?;
    let router_bias = f32_or_empty(op.router_bias.as_ref())?;
    let gate_up_bias = f32_or_empty(op.gate_up.bias.as_ref())?;
    let down_bias = f32_or_empty(op.down.bias.as_ref())?;
    let pre_ffn_norm = store.load(&layer.pre_ffn_norm.weight)?;
    let gate_rule = larql_compute::MoeGateRule::from_arch(op.gate_policy, op.activation);

    let scratch = larql_compute_metal::MoeScratch::new_public_with_format(
        gpu,
        op.top_k,
        hidden,
        inter,
        larql_compute::QuantFormat::MXFP4,
        hidden,
    );
    // Build the descriptor table from a temporary `MoeLayerWeights`
    // borrowing the freshly-loaded storage, before that storage moves
    // into `RoutedLayer` — the table keeps only region buffers, no borrow.
    let table = {
        use larql_compute::{
            MoeExpertScales, MoeFusedRowLayout, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
        };
        let moe = larql_compute::MoeLayerWeights {
            experts_gate_up: expert_slices(&gate_up_blocks, gu_expert_bytes, experts),
            experts_down: expert_slices(&down_blocks, dn_expert_bytes, experts),
            routing_policy: MoeRoutingPolicy::top_k_then_softmax(),
            weight_layout: MoeWeightLayout::unpadded(),
            expert_scales: MoeExpertScales::Paired {
                gate_up: expert_slices(&gate_up_scales, gu_scale_bytes, experts),
                down: expert_slices(&down_scales, dn_scale_bytes, experts),
            },
            fused_row_layout: MoeFusedRowLayout::Interleaved,
            expert_data_format: QuantFormat::MXFP4,
            router_proj: &router_proj,
            router_bias: &router_bias,
            experts_gate_up_bias: &gate_up_bias,
            experts_down_bias: &down_bias,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_ffn_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: experts,
            top_k: op.top_k,
            intermediate_size: inter,
            gate_rule,
        };
        if !gpu.lowering_moe_supported(&moe, &scratch) {
            return Err(VindexError::Parse(format!(
                "layer {}: the descriptor MoE path does not support this routed layer \
                 (format/policy/geometry) — refusing before encode",
                layer.layer
            )));
        }
        gpu.lowering_moe_descriptor(layer.layer, &moe, inter, hidden)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "layer {}: expert operands did not resolve inside their registered regions",
                    layer.layer
                ))
            })?
    };
    Ok(RoutedLayer {
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
        router_proj,
        router_bias,
        gate_up_bias,
        down_bias,
        pre_ffn_norm,
        gu_expert_bytes,
        gu_scale_bytes,
        dn_expert_bytes,
        dn_scale_bytes,
        experts,
        top_k: op.top_k,
        inter,
        gate_rule,
        table,
        scratch,
        eps: layer.pre_ffn_norm.eps as f32,
    })
}

/// The dense FFN op of a layer the lowering has already admitted (routed
/// layers are refused in `new`, so this only fails on a plan that changed
/// under us).
fn dense_ffn(layer: &LayerPlan) -> Result<&FfnOp, VindexError> {
    layer.ffn.dense().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {} carries a routed FFN the lowering does not execute (A-9.4)",
            layer.layer
        ))
    })
}

fn argmax_of(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, v)| {
                if *v > bv {
                    (i, *v)
                } else {
                    (bi, bv)
                }
            },
        )
        .0 as u32
}

/// Run the plan through the lowering and report the final position's
/// logits, in the same shape `run_exec`'s other arms do.
pub(super) fn run_lowered(
    args: &super::ExecArgs,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    formats: WeightFormats,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let gpu = MetalBackend::new().ok_or("no Metal device available for --backend metal-lowered")?;
    let total = tokens.len() + args.generate.unwrap_or(0);
    let loading = std::time::Instant::now();
    let mut keep = Vec::new();
    let mut session = LoweredSession::new(&gpu, plan, store, formats, total.max(1), &mut keep)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    eprintln!("weights resident in {load_seconds:.1} s");
    if let Some((rows, cols)) = session.head_geometry() {
        eprintln!("head geometry: [{rows}, {cols}]");
    }
    eprintln!(
        "plan: {} rope base(s), final norm {}",
        session.rope_bases(),
        if session.has_final_norm() {
            "present"
        } else {
            "absent"
        }
    );

    let prompt_started = std::time::Instant::now();
    let mut logits: Option<Vec<f32>> = None;
    for &token in tokens {
        logits = session.step(token)?;
    }
    // ── decode, kept strictly separate from prefill ─────────────────
    let mut decode_ms: Vec<f64> = Vec::new();
    let mut generated: Vec<u32> = Vec::new();
    if let Some(n) = args.generate {
        let mut next = logits
            .as_ref()
            .map(|l| argmax_of(l))
            .ok_or("plan carries no output head — cannot generate")?;
        for _ in 0..n {
            generated.push(next);
            let started = std::time::Instant::now();
            let l = session.step(next)?.ok_or("plan carries no output head")?;
            decode_ms.push(started.elapsed().as_secs_f64() * 1e3);
            next = argmax_of(&l);
            logits = Some(l);
        }
    }

    let prompt_seconds = prompt_started.elapsed().as_secs_f64();
    if session.ablation_active() {
        println!("ABLATED RUN — numbers are wrong by construction; timing only");
    }
    println!("engine: vindex3-metal-lowered-{label}");
    println!("weights loaded: {load_seconds:.1} s");
    println!(
        "prompt: {} tokens in {prompt_seconds:.1} s ({:.0} ms/token)",
        tokens.len(),
        prompt_seconds * 1e3 / tokens.len().max(1) as f64,
    );
    if !decode_ms.is_empty() {
        let mut sorted = decode_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        // Steady state = the second half, so warmup and first-touch
        // residency do not flatter or penalise the median.
        let steady = &decode_ms[decode_ms.len() / 2..];
        let steady_mean = steady.iter().sum::<f64>() / steady.len() as f64;
        println!("decode tokens: {}", decode_ms.len());
        println!("first token: {:.0} ms", decode_ms[0]);
        println!("decode p50: {:.0} ms  p95: {:.0} ms", pct(0.50), pct(0.95));
        println!(
            "steady (last half): {:.0} ms/token ({:.3} tok/s)",
            steady_mean,
            1000.0 / steady_mean
        );
        println!("generated ids: {generated:?}");
        // Which attention kernel actually ran — the seqpar port is judged
        // by this witness, not inferred from a throughput number.
        {
            use std::sync::atomic::Ordering;
            let serial =
                larql_compute_metal::route_witness::LOWERED_ATTEND_SERIAL.load(Ordering::Relaxed);
            let seqpar =
                larql_compute_metal::route_witness::LOWERED_ATTEND_SEQPAR.load(Ordering::Relaxed);
            println!("attention dispatches: serial {serial}  seqpar {seqpar}");
        }
    }
    match &logits {
        Some(l) => {
            let (best, value) =
                l.iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::MIN),
                        |(bi, bv), (i, v)| {
                            if *v > bv {
                                (i, *v)
                            } else {
                                (bi, bv)
                            }
                        },
                    );
            println!("logits: {}, argmax {best} ({value:+.4})", l.len());
            if let Some(path) = &args.logit_dump {
                use std::io::Write;
                let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
                for v in l {
                    f.write_all(&v.to_le_bytes())?;
                }
                f.flush()?;
                println!("wrote [{}] f32 to {}", l.len(), path.display());
            }
        }
        None => println!("logits: none (plan carries no output head)"),
    }
    Ok(())
}
