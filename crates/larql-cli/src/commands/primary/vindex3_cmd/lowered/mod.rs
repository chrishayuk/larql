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
use larql_vindex::format::vindex3::opplan::exec::weights::LoadedWeight;
use larql_vindex::format::vindex3::opplan::{ComponentOpPlan, LayerPlan};

/// One matrix operand, resident on the device.
mod dump;
mod resident;
mod routed;

use dump::dump_lowered;
use resident::{
    resident_matrix, resident_norm, resident_vector, rope_inv_freq_table, rope_table_key, Ablation,
};
use routed::{build_ffn, FfnResident};

pub(super) struct DeviceMatrix {
    /// `scales` is unused for f16; the representation is what the plan's
    /// per-class policy asked for, not something inferred here.
    pub(super) packed: DeviceBuffer,
    pub(super) scales: DeviceBuffer,
    pub(super) tensor_scale: f32,
    pub(super) format: WeightFormat,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl DeviceMatrix {
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
        self.step_impl(token, None)
    }

    /// One step, capturing the embedding row and every layer's output for
    /// this position — the per-layer planes a `shannon layer-diff` reads.
    /// `layers_out[i]` is layer `i`'s post-FFN-residual hidden state.
    pub fn step_capturing(
        &mut self,
        token: u32,
    ) -> Result<(Option<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>), VindexError> {
        let mut embedding = Vec::new();
        let mut layers_out = Vec::new();
        let logits = self.step_impl(token, Some((&mut embedding, &mut layers_out)))?;
        Ok((logits, embedding, layers_out))
    }

    fn step_impl(
        &mut self,
        token: u32,
        capture: Option<(&mut Vec<f32>, &mut Vec<Vec<f32>>)>,
    ) -> Result<Option<Vec<f32>>, VindexError> {
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
        let capturing = capture.is_some();
        // Per-layer capture buffers (hidden-sized), read back after the
        // command buffer completes — a copy inside the stream, never a
        // mid-stream readback.
        let captures: Vec<DeviceBuffer> = if capturing {
            (0..self.plan.layers.len())
                .map(|_| self.gpu.lowering_scratch(self.hidden))
                .collect()
        } else {
            Vec::new()
        };
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

        let checkpoints: Vec<larql_compute_metal::lowering::stack::Checkpoint> = captures
            .iter()
            .enumerate()
            .map(
                |(i, buf)| larql_compute_metal::lowering::stack::Checkpoint {
                    after_layer: i,
                    into: buf,
                },
            )
            .collect();
        let cmd = self.gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let h_final = self
            .gpu
            .encode_stack(enc, &h_in, &layers, &scratch, &checkpoints);

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
        if let Some((embedding, layers_out)) = capture {
            *embedding = h0;
            for buf in &captures {
                layers_out.push(
                    self.gpu
                        .lowering_readback(buf, self.hidden)
                        .ok_or_else(|| VindexError::Parse("capture readback failed".into()))?,
                );
            }
        }
        for buf in captures {
            self.gpu.recycle_lowering_scratch(buf);
        }
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

    // ── per-layer dump: teacher-force the given tokens, capturing every
    //    layer's output per position into [seq, hidden] planes a
    //    `shannon layer-diff` reads (the lowered arm of the A-9.5 chain).
    if let Some(dir) = &args.dump_layers {
        return dump_lowered(&mut session, tokens, plan, &args.container, label, dir);
    }

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
