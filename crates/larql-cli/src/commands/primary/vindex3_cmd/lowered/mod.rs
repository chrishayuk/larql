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
use larql_compute_metal::lowering::attention::{
    AttnShape, AttnWeights, LoweredPosition, QkNormWeights,
};
use larql_compute_metal::lowering::ffn::{FfnActivation, FfnShape, FfnWeights};
use larql_compute_metal::lowering::stack::{
    HybridFfnLowering, LayerFfnLowering, LayerLowering, RoutedFfnLowering,
};
use larql_compute_metal::lowering::{DeviceBuffer, LoweredMatrix, PostNorm};
use larql_compute_metal::MetalBackend;
use larql_models::config::PositionPolicy;
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::graph::policy::AttentionSpan;
use larql_vindex::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::LoadedWeight;
use larql_vindex::format::vindex3::opplan::{ComponentOpPlan, LayerPlan};

/// One matrix operand, resident on the device.
mod dump;
mod profile;
mod resident;
mod routed;
mod run;
mod step;
#[cfg(test)]
mod tests;

pub(super) use run::run_lowered;

use profile::{StageBytes, StageLedger};
use resident::{
    resident_attn, resident_matrix, resident_norm, resident_vector, rope_inv_freq_table,
    rope_table_key, Ablation,
};
use routed::{build_ffn, FfnResident};

pub(super) struct DeviceMatrix {
    /// `scales` is unused for f16; the representation is what the plan's
    /// per-class policy asked for, not something inferred here.
    pub(super) packed: DeviceBuffer,
    pub(super) scales: DeviceBuffer,
    /// Byte offsets of this matrix's rows inside `packed`/`scales` —
    /// non-zero when the matrix is a slice of a shared allocation (the
    /// QKV packing rung). NVFP4 only; the other formats are always
    /// whole-buffer residents.
    pub(super) packed_offset: u64,
    pub(super) scales_offset: u64,
    /// Bytes a matvec over this matrix reads — its own rows, not the
    /// (possibly shared) allocation the buffers measure.
    pub(super) read_bytes: usize,
    pub(super) tensor_scale: f32,
    pub(super) format: WeightFormat,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl DeviceMatrix {
    /// Bytes a matvec over this matrix reads: packed codes, plus scales
    /// for the block formats (f16 carries an empty scales buffer).
    pub(super) fn bytes(&self) -> usize {
        self.read_bytes
    }

    pub(super) fn as_lowered(&self) -> LoweredMatrix<'_> {
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
                packed_offset: self.packed_offset,
                scales: &self.scales,
                scales_offset: self.scales_offset,
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
    /// Weighted per-head Q/K norm weights and their offset, when the plan
    /// carries the op.
    qk_norm: Option<(DeviceBuffer, DeviceBuffer, f32)>,
    /// This layer's rotary table key (into `LoweredSession::inv_freq`);
    /// `None` on a NoPE layer.
    rope_key: Option<u64>,
    /// The layer's output scalar, when the plan carries one.
    layer_scale: Option<f32>,
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
    ablate: Ablation,
    /// `Some` while `--profile` is recording decode tokens.
    ledger: Option<StageLedger>,
    /// GPU span of the most recent step's command buffer, in ms — the
    /// token's device time, so wall minus this is host time.
    last_gpu_ms: f64,
    /// Host time the most recent step spent encoding the command buffer
    /// (before commit), in ms — overlapped with the previous token's GPU
    /// execution by `step`, so only the first token pays it on the wall.
    last_encode_ms: f64,
    /// The next position's command buffer, encoded ahead of its input
    /// (see `step.rs`).
    prepared: Option<step::PreparedStep>,
    /// KV capacity in positions; nothing is encoded past it.
    max_positions: usize,
    /// Device scratch for the head's argmax: block partials (values,
    /// indices) and TWO one-u32 results, alternated by position parity —
    /// with commit-ahead (1c) step t+1 executes while the host still
    /// reads step t's id, so they must not share the output word.
    /// `None` without a head.
    argmax: Option<[DeviceBuffer; 4]>,
    /// The embedding table resident on the device (zero-copy over the
    /// host allocation), for the 1c gather path. `None` when the plan
    /// carries a judged embedding norm — the host computes that in f64,
    /// which the f32 kernel cannot reproduce, so those plans keep the
    /// host embed.
    device_embed: Option<DeviceBuffer>,
    /// The id the device argmax produced for the most recent completed
    /// step; a decode step whose input token equals it can gather the
    /// embedding on the device instead of uploading a host row.
    last_device_id: Option<u32>,
    /// Set by `begin_decode`: every following step continues from the
    /// device argmax, so look-ahead steps may gather their embedding on
    /// the device and be committed before their predecessor completes.
    /// Never set during the prompt — a prompt look-ahead's token is the
    /// caller's, not the argmax's, and a committed wrong-token step
    /// would execute (and burn GPU time) before being discarded.
    decode_chain: bool,
}

/// Set to keep the argmax on the host (full-logits readback + scan) —
/// the control arm for the device argmax, not a production setting.
const HOST_ARGMAX_ENV: &str = "LARQL_LOWERED_HOST_ARGMAX";

/// Stage runs one profiled token may hold before attribution stops. The
/// device refuses a timestamp sample buffer above 4096 samples (two per
/// run — `examples/stage_profiler_probe.rs`), and 2048 runs covers the
/// finest class split (≤ 10 per layer) on a 200-layer stack.
const PROFILE_MAX_STAGE_RUNS: usize = 2048;

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
                .as_ref()
                .and_then(|f| f.dense())
                .is_some_and(|f| !matches!(f.gate_policy, larql_models::ExpertGatePolicy::Gated))
        }) {
            return Err(VindexError::Parse(format!(
                "layer {} carries {:?}, which the Metal lowering does not execute yet (A-9.4); \
                 refusing rather than lowering it as plain gating",
                l.layer,
                l.ffn
                    .as_ref()
                    .and_then(|f| f.dense())
                    .map(|f| f.gate_policy)
            )));
        }
        // Gemma 4's semantics are lowered (G4.3): K≡V binds the K matrix
        // as V, the V norm and the weighted Q/K norms ride the served
        // norm kernels, the proportional partial rotary rides a per-layer
        // table, the hybrid FFN is composed in `encode_stack`. What the
        // lowering still has no kernel for is refused, typed, here: a
        // rotary-width partial rotary (a prefix rotated as its own block)
        // and a gate activation other than SiLU / tanh-GELU.
        // No DeltaNet kernel exists on this path. Refuse the whole plan
        // rather than lower the 16 softmax layers of a 64-layer hybrid and
        // silently drop the other 48.
        if let Some(l) = plan.layers.iter().find(|l| l.attention.softmax().is_none()) {
            return Err(VindexError::Parse(format!(
                "layer {} carries `{}`, which this lowering has no kernel for; refusing",
                l.layer,
                l.attention.declared_name(),
            )));
        }
        if let Some(l) = plan.layers.iter().find(|l| {
            matches!(
                l.attention.softmax().map(|op| &op.position),
                // M-RoPE joins the refusal on the same ground: its
                // rotary block is prefix-shaped, and its per-slot axis
                // assignment is a second thing the rope kernel does not
                // express.
                Some(
                    PositionPolicy::PartialRope {
                        basis: larql_models::config::RotaryFrequencyBasis::RotaryWidth,
                        ..
                    } | PositionPolicy::MRope { .. }
                )
            )
        }) {
            return Err(VindexError::Parse(format!(
                "layer {} carries {:?}, whose prefix-block rotation the rope kernel does not \
                 express; refusing rather than rotating the whole head",
                l.layer,
                l.attention.softmax().map(|op| &op.position)
            )));
        }
        if let Some(l) = plan
            .layers
            .iter()
            .find(|l| l.layer_scale.is_some() && l.ffn.as_ref().and_then(|f| f.hybrid()).is_none())
        {
            return Err(VindexError::Parse(format!(
                "layer {} carries a layer scalar on a non-hybrid FFN, which the stack encoder \
                 applies only on the hybrid arm today; refusing",
                l.layer
            )));
        }
        for l in &plan.layers {
            let activation = match &l.ffn {
                Some(larql_vindex::format::vindex3::opplan::LayerFfn::Dense(op)) => {
                    Some(op.activation)
                }
                Some(larql_vindex::format::vindex3::opplan::LayerFfn::Hybrid(op)) => {
                    Some(op.dense.activation)
                }
                Some(larql_vindex::format::vindex3::opplan::LayerFfn::Routed(_)) | None => None,
            };
            if let Some(activation) = activation {
                ffn_activation(activation)
                    .map_err(|e| VindexError::Parse(format!("layer {}: {e}", l.layer)))?;
            }
        }
        let embedding = plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("plan carries no embedding op".into()))?;
        let embed_table = store.load(&embedding.table)?;
        let hidden = embed_table.len() / embedding.vocab_size;

        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let a = layer.attention.softmax().unwrap_or_else(|| {
                panic!(
                    "layer {} is not softmax; the lowering refused this plan in `new`",
                    layer.layer
                )
            });
            let kv_rows = a.num_kv_heads * a.head_dim;
            let zeros = vec![0.0f32; max_positions * kv_rows];
            // On a K≡V layer the plan's `v` IS the K operand: the same
            // bytes load through the same cache, so the V projection
            // binds the K matrix — the raw K projection lands in the V
            // slot before the key's own norm and rotation.
            // Q, K, V and O as slices of one allocation, in touch order,
            // where format and alignment admit it; otherwise four
            // separate residents, unchanged.
            let [q_m, k_m, v_m, o_m] = resident_attn(
                gpu,
                store,
                [&a.q, &a.k, &a.v, &a.o],
                formats.attention,
                keep,
            )?;
            layers.push(LayerResident {
                q: q_m,
                k: k_m,
                v: v_m,
                o: o_m,
                qk_norm: match &a.qk_norm {
                    Some(qk) => {
                        let q = resident_vector(gpu, store, Some(&qk.q))?.expect("q norm weight");
                        let k = resident_vector(gpu, store, Some(&qk.k))?.expect("k norm weight");
                        Some((q, k, qk.weight_offset))
                    }
                    None => None,
                },
                rope_key: rope_table_key(&a.position, a.head_dim),
                layer_scale: match &layer.layer_scale {
                    Some(op) => Some(store.load(op).and_then(|v| {
                        larql_vindex::format::vindex3::opplan::exec::layer_scalar_of(&v)
                    })?),
                    None => None,
                },
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
                // The Metal trunk applies a norm before attention. A
                // post-norm stack has none, and this path refuses rather
                // than lowering an identity norm that would read as one.
                pre_attn_norm: resident_norm(
                    gpu,
                    store,
                    layer.pre_attention_norm.as_ref().ok_or_else(|| {
                        VindexError::Parse(format!(
                            "layer {} carries no pre-attention norm (post-norm placement); the \
                             Metal trunk has no lowering for it",
                            layer.layer
                        ))
                    })?,
                )?
                .0,
                post_attn_norm: match &layer.post_attention_norm {
                    Some(op) => Some(resident_norm(gpu, store, op)?),
                    None => None,
                },
                pre_ffn_norm: resident_norm(
                    gpu,
                    store,
                    layer.pre_ffn_norm.as_ref().ok_or_else(|| {
                        VindexError::Parse(format!(
                            "layer {} carries no pre-FFN norm (mixer-only); the Metal \
                             lowering has no arm for it",
                            layer.layer
                        ))
                    })?,
                )?
                .0,
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
            .filter_map(|l| l.attention.softmax())
            .map(|op| op.num_q_heads * op.head_dim)
            .max()
            .unwrap_or(hidden);
        let max_inter = plan
            .layers
            .iter()
            .filter_map(|l| {
                let ffn = l.ffn.as_ref()?;
                ffn.dense()
                    .map(|f| f.intermediate_size)
                    .or_else(|| ffn.hybrid().map(|h| h.dense.intermediate_size))
            })
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
        let mut scratch: Vec<DeviceBuffer> =
            sizes.iter().map(|n| gpu.lowering_scratch(*n)).collect();
        // A hybrid layer's own intermediates (slots 18..24), and a zero
        // buffer for the expert combine's residual input.
        let has_hybrid = plan
            .layers
            .iter()
            .any(|l| l.ffn.as_ref().and_then(|f| f.hybrid()).is_some());
        if has_hybrid {
            for _ in 0..larql_compute_metal::lowering::stack::StackScratch::HYBRID_BUFFERS {
                scratch.push(gpu.lowering_scratch(hidden));
            }
            let zero = vec![0.0f32; hidden];
            scratch.push(
                gpu.lowering_upload(&zero)
                    .ok_or_else(|| VindexError::Parse("zero buffer upload failed".into()))?,
            );
        }

        // One inverse-frequency table per distinct rotary policy in the
        // plan — keyed on (theta, yarn-or-plain), so a YaRN layer's ramped
        // frequencies and a plain layer's `theta^(-2i/d)` never collide on
        // theta alone. The table matches the interpreter's exactly: plain
        // rope from `rope_rotate`, YaRN from `kernels::yarn_frequencies`.
        let mut inv_freq: HashMap<u64, DeviceBuffer> = HashMap::new();
        for layer in &plan.layers {
            let a = layer.attention.softmax().unwrap_or_else(|| {
                panic!(
                    "layer {} is not softmax; the lowering refused this plan in `new`",
                    layer.layer
                )
            });
            let key = rope_table_key(&a.position, a.head_dim);
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

        // The embedding table as a device buffer over the same host
        // floats — a row lookup on the GPU reads only the sampled row's
        // 4·hidden bytes, so residency does not change. Refused when the
        // plan judges a weightless embedding norm (host f64 semantics).
        let device_embed = (plan.embedding.as_ref().is_some_and(|e| e.norm.is_none())).then(|| {
            // SAFETY: an f32 slice viewed as bytes, same length; the Vec
            // lives in `Self` for the session, outliving the buffer's use.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    embed_table.as_ptr() as *const u8,
                    std::mem::size_of_val(embed_table.as_slice()),
                )
            };
            gpu.lowering_weight(bytes)
        });

        // Argmax scratch sized for the head's vocabulary. The control arm
        // (`LARQL_LOWERED_HOST_ARGMAX=1`) keeps the argmax on the host so
        // the device kernel can be A/B'd under one power state.
        let host_argmax = std::env::var_os(HOST_ARGMAX_ENV).is_some();
        let argmax = (vocab > 0 && !host_argmax).then(|| {
            use larql_compute_metal::lowering::head::argmax_partials;
            let parts = argmax_partials(vocab);
            [
                gpu.lowering_scratch(parts),
                gpu.lowering_scratch(parts),
                gpu.lowering_scratch(1),
                gpu.lowering_scratch(1),
            ]
        });
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
            ledger: None,
            last_gpu_ms: 0.0,
            last_encode_ms: 0.0,
            prepared: None,
            max_positions,
            argmax,
            device_embed,
            last_device_id: None,
            decode_chain: false,
        })
    }

    fn layer_lowering<'b>(
        &'b self,
        plan_layer: &'b LayerPlan,
        r: &'b LayerResident,
        t: usize,
    ) -> LayerLowering<'b> {
        let a = plan_layer.attention.softmax().unwrap_or_else(|| {
            panic!(
                "layer {} is not softmax; the lowering refused this plan in `new`",
                plan_layer.layer
            )
        });
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
                qk_norm: r.qk_norm.as_ref().filter(|_| !self.ablate.no_qk_norm).map(
                    |(q, k, offset)| QkNormWeights {
                        q,
                        k,
                        weight_offset: *offset,
                    },
                ),
                norm_weight: &r.pre_attn_norm,
                post_norm: post(&r.post_attn_norm, &self.scratch[7])
                    .filter(|_| !self.ablate.no_post_norms),
            },
            attn_shape: AttnShape {
                hidden: self.hidden,
                num_q_heads: a.num_q_heads,
                num_kv_heads: a.num_kv_heads,
                head_dim: a.head_dim,
                norm_eps: plan_layer.declared_norm_eps as f32,
                // The weight offset of the norm that conditions the
                // attention input. Under post-norm placement no norm does,
                // so the offset that would scale nothing is the identity.
                norm_weight_offset: plan_layer
                    .pre_attention_norm
                    .as_ref()
                    .map_or(0.0, |n| n.weight_offset),
                // The component's declared epsilon, which QK norm runs at.
                // Read from the layer's own field rather than off a norm
                // site that a post-norm stack does not carry.
                qk_norm_eps: plan_layer.declared_norm_eps as f32,
                parameter_free_q: a.parameter_free_qk_norm.q && !self.ablate.no_qk_norm,
                parameter_free_k: a.parameter_free_qk_norm.k && !self.ablate.no_qk_norm,
                parameter_free_v: a.parameter_free_qk_norm.v && !self.ablate.no_qk_norm,
                query_scale: a
                    .query_scale
                    .map(|s| s as f32)
                    .filter(|_| !self.ablate.no_query_scale),
                score_scale: a.score_scale as f32,
                position: match a.position {
                    _ if self.ablate.no_rope => LoweredPosition::None,
                    PositionPolicy::Rope { theta } => LoweredPosition::Rope { theta },
                    // Refused above, before the session exists.
                    PositionPolicy::MRope { .. } => {
                        unreachable!("M-RoPE is refused before the session is built")
                    }
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
                    // Llama-3 rides the same shared table as YaRN — the
                    // per-layer `inv_freq` built in `new` — but at unit
                    // amplitude: the family adjusts frequencies only.
                    // Written as an explicit 1.0 rather than reusing
                    // YaRN's arm, so an amplitude can never be inherited
                    // by a family that does not define one.
                    PositionPolicy::Llama3 { theta, .. } => LoweredPosition::Scaled {
                        theta,
                        amplitude: 1.0,
                    },
                    // No lowering exists for a relative scheme. It
                    // lowers to `None` — no rotation — and the executor
                    // refuses rather than running it unpositioned, so the
                    // absence is never mistaken for NoPE downstream.
                    PositionPolicy::Relative { .. } => LoweredPosition::None,
                    PositionPolicy::None => LoweredPosition::None,
                    // The proportional table (zeros above the fraction)
                    // rides this layer's own inv_freq at unit amplitude;
                    // the rotary-width basis was refused in `new`.
                    PositionPolicy::PartialRope { theta, .. } => LoweredPosition::Scaled {
                        theta,
                        amplitude: 1.0,
                    },
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
                            .as_ref()
                            .and_then(|f| f.dense())
                            .map_or(self.hidden, |f| f.intermediate_size),
                        norm_eps: plan_layer
                            .pre_ffn_norm
                            .as_ref()
                            .expect("dense resident implies a pre-FFN norm")
                            .eps as f32,
                        norm_weight_offset: plan_layer
                            .pre_ffn_norm
                            .as_ref()
                            .expect("dense resident implies a pre-FFN norm")
                            .weight_offset,
                        activation: plan_layer
                            .ffn
                            .as_ref()
                            .and_then(|f| f.dense())
                            .map_or(FfnActivation::Silu, |f| {
                                ffn_activation(f.activation).expect("checked in `new`")
                            }),
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
                FfnResident::Hybrid(h) => {
                    let op = plan_layer
                        .ffn
                        .as_ref()
                        .and_then(|f| f.hybrid())
                        .expect("resident matches the plan");
                    LayerFfnLowering::Hybrid(Box::new(HybridFfnLowering {
                        dense: FfnWeights {
                            gate: h.gate.as_lowered(),
                            up: h.up.as_lowered(),
                            down: h.down.as_lowered(),
                            norm_weight: &r.pre_ffn_norm,
                            // The hybrid applies the layer's post-FFN norm
                            // itself, after summing the branches.
                            post_norm: None,
                        },
                        dense_shape: FfnShape {
                            hidden: self.hidden,
                            intermediate: op.dense.intermediate_size,
                            norm_eps: plan_layer
                                .pre_ffn_norm
                                .as_ref()
                                .expect("hybrid resident implies a pre-FFN norm")
                                .eps as f32,
                            norm_weight_offset: plan_layer
                                .pre_ffn_norm
                                .as_ref()
                                .expect("hybrid resident implies a pre-FFN norm")
                                .weight_offset,
                            activation: ffn_activation(op.dense.activation)
                                .expect("checked in `new`"),
                        },
                        routed: RoutedFfnLowering {
                            moe: h.routed.moe(),
                            scratch: &h.routed.scratch,
                            table: &h.routed.table,
                            eps: h.routed.eps,
                        },
                        router_conditioning: &h.router_conditioning,
                        per_expert_scale: &h.per_expert_scale,
                        pre_experts_norm: &h.pre_experts_norm,
                        post_dense_norm: &h.post_dense_norm,
                        post_experts_norm: &h.post_experts_norm,
                        branch_norm_eps: h.branch_norm_eps,
                        branch_norm_weight_offset: h.branch_norm_weight_offset,
                        post_ffn_norm: post(&r.post_ffn_norm, &self.scratch[14])
                            .filter(|_| !self.ablate.no_post_norms),
                        layer_scale: r.layer_scale,
                    }))
                }
            },
            k_cache: &r.k_cache,
            v_cache: &r.v_cache,
            inv_freq: r
                .rope_key
                .and_then(|k| self.inv_freq.get(&k))
                .unwrap_or(&self.scratch[0]),
        }
    }

    /// Matrix geometry the loader saw, for diagnostics.
    /// GPU span of the most recent step, in ms.
    pub fn last_gpu_ms(&self) -> f64 {
        self.last_gpu_ms
    }

    /// Host encode time of the most recent step, in ms.
    pub fn last_encode_ms(&self) -> f64 {
        self.last_encode_ms
    }

    /// Start recording per-stage GPU time for every following step.
    pub fn start_profile(&mut self) {
        // A step encoded ahead of this call carries no sampler; drop it
        // so the first profiled token is encoded under the profiler.
        if let Some(p) = self.prepared.take() {
            self.discard(p);
        }
        self.ledger = Some(StageLedger {
            bytes: self.stage_bytes(),
            ..Default::default()
        });
    }

    /// The recorded ledger, rendered; `None` if profiling never started.
    pub fn profile_report(&self) -> Option<Vec<String>> {
        self.ledger.as_ref().map(|l| l.render())
    }

    /// Bytes one token reads per stage class, from the resident
    /// operands.
    fn stage_bytes(&self) -> StageBytes {
        let mut b = StageBytes::default();
        for l in &self.layers {
            b.attn_proj += l.q.bytes() + l.k.bytes() + l.v.bytes();
            b.attn_out += l.o.bytes();
            if let Some(g) = &l.gate {
                b.attn_proj += g.bytes();
            }
            let (dense, experts) = l.ffn.bytes_per_token();
            b.dense_ffn += dense;
            b.experts += experts;
        }
        if let Some(h) = &self.head {
            b.head = h.bytes();
        }
        b
    }

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

/// First hybrid scratch slot: the 18 stack slots precede it (slots 16/17
/// are the head's vocabulary-sized pair).
const HYBRID_SCRATCH_BASE: usize = 18;

/// The lowering's gate activation for the plan's, or why there is none.
fn ffn_activation(
    activation: larql_models::config::Activation,
) -> Result<FfnActivation, VindexError> {
    use larql_models::config::Activation;
    match activation {
        Activation::Silu => Ok(FfnActivation::Silu),
        Activation::GeluTanh => Ok(FfnActivation::GeluTanh),
        other => Err(VindexError::Parse(format!(
            "the lowering has no gate/up kernel for activation {other:?}; refusing"
        ))),
    }
}
