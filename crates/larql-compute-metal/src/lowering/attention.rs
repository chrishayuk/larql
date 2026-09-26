//! Lowering a plan's attention op into one encoder (VINDEX3-G6b-3).
//!
//! The delicate fragment. Unlike the FFN, attention is an **ordered**
//! program whose steps approximately commute, so a lowering can contain
//! every operation, produce plausible numbers, and still represent a
//! different model. The order below is the interpreter's
//! `condition_qk_in_place`, transcribed rather than reconstructed:
//!
//! ```text
//! h ─ pre-attn norm ─┬─ Q proj ─ param-free QK norm ─ query scale ─ RoPE ─┐
//!                    ├─ K proj ─ param-free QK norm ─────────────  RoPE ──┤ (into KV cache)
//!                    ├─ V proj ───────────────────────────────────────────┤ (into KV cache)
//!                    └─ gate proj ────────────────────────┐               │
//!                                                          │      attention
//!                                                          │          │
//!                                            sigmoid gate ─┴──────────┘
//!                                                          │
//!                                            o_proj ─ post_attn_norm ─ residual ─ h'
//! ```
//!
//! **Query scale applies to Q only, after QK norm and before RoPE.** All
//! three touch Q, and swapping any pair changes the model while leaving
//! magnitudes plausible — the parity test carries an explicit ordering
//! control for exactly this.
//!
//! K and V project **directly into their KV-cache slots** rather than
//! into scratch that is later copied: the cache is `[T, num_kv,
//! head_dim]` position-major, so the current position's slot is a plain
//! byte offset, and the in-place QK norm and RoPE then operate on the
//! cache through the same offset. Removing the copy also removes the
//! chance of the cached K diverging from the K that was normed.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::profile::{Stage, StageEncoders};

use super::{
    nvfp4_residual_fusion_enabled, nvfp4_segment, LoweredMatrix, MatmulRowsTarget, MatvecOperands,
    MatvecTarget, Nvfp4Segment, PostNorm,
};
use crate::MetalBackend;

/// Position encoding for this layer, from its own policy entry — never a
/// model-wide default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoweredPosition {
    /// Rotary at this base, unit amplitude.
    Rope { theta: f64 },
    /// Rotary at frequencies the caller's `inv_freq` table already
    /// carries (YaRN's ramped blend), with this amplitude on `cos`/`sin`
    /// — the part of YaRN that rescales every logit at every position.
    Scaled { theta: f64, amplitude: f32 },
    /// The layer attends position-agnostically (NoPE).
    None,
}

impl LoweredPosition {
    /// The `cos`/`sin` scalar this policy applies; `None` for a layer
    /// that does not rotate.
    fn amplitude(self) -> Option<f32> {
        match self {
            Self::Rope { .. } => Some(1.0),
            Self::Scaled { amplitude, .. } => Some(amplitude),
            Self::None => None,
        }
    }
}

/// Everything attention reads.
pub struct AttnWeights<'a> {
    pub q: LoweredMatrix<'a>,
    pub k: LoweredMatrix<'a>,
    pub v: LoweredMatrix<'a>,
    pub o: LoweredMatrix<'a>,
    /// The judged attention output gate. `None` = no gate op — which is
    /// a different claim from a gate that happens to be near 1.
    pub gate: Option<LoweredMatrix<'a>>,
    /// Q/K/V/O projection biases (f32), from the plan's operands: added
    /// right after each projection — before QK-norm/RoPE for Q and K,
    /// before the cache holds V, after `o` before the residual. `None`
    /// = the plan carries no bias for that projection.
    pub q_bias: Option<&'a Buffer>,
    pub k_bias: Option<&'a Buffer>,
    pub v_bias: Option<&'a Buffer>,
    pub o_bias: Option<&'a Buffer>,
    /// Per-query-head attention-sink logits (f32), when the plan's
    /// judged sink semantics apply; `None` = ordinary softmax.
    pub sinks: Option<&'a Buffer>,
    /// Weighted per-head Q/K norms (Gemma `q_norm` / `k_norm`), applied
    /// after the projections and before the query scale and rotation,
    /// with the plan's weight offset. `None` = the op is absent.
    pub qk_norm: Option<QkNormWeights<'a>>,
    /// Pre-attention norm weight (f32).
    pub norm_weight: &'a Buffer,
    /// The post-attention norm, under four-norm placement. `None` =
    /// absent.
    pub post_norm: Option<PostNorm<'a>>,
}

/// Caller-owned device scratch and cache.
pub struct AttnScratch<'a> {
    /// `hidden` floats.
    pub normed: &'a Buffer,
    /// `num_q_heads * head_dim` floats.
    pub q: &'a Buffer,
    /// `[T, num_kv_heads, head_dim]` — K and V caches, written in place.
    pub k_cache: &'a Buffer,
    pub v_cache: &'a Buffer,
    /// `num_q_heads * head_dim` floats each.
    pub gate: &'a Buffer,
    pub concat: &'a Buffer,
    pub gated: &'a Buffer,
    /// `hidden` floats — o_proj output, before the residual.
    pub attn_out: &'a Buffer,
    /// `head_dim / 2` floats, host-computed to match the interpreter's
    /// `theta^(-2i/head_dim)`.
    pub inv_freq: &'a Buffer,
    /// SPLITK-1 partials. `None` = split-K is never dispatched and the
    /// planner's seqpar/serial choice runs unchanged.
    pub splitk: Option<&'a crate::ops::kv_splitk::SplitKScratch<'a>>,
}

/// The weighted QK-norm operands: one `[head_dim]` vector each.
pub struct QkNormWeights<'a> {
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    /// Centred-norm convention (`1 + w`); a plan fact.
    pub weight_offset: f32,
}

/// Geometry and judged semantics, straight off the plan.
#[derive(Clone)]
pub struct AttnShape {
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub norm_weight_offset: f32,
    pub qk_norm_eps: f32,
    pub parameter_free_q: bool,
    pub parameter_free_k: bool,
    /// Parameter-free per-head RMS norm on V (Gemma 4 `v_norm`), applied
    /// to the raw value projection in its cache slot — before anything
    /// reads it, and on a K≡V layer before the key's own norm/rotation
    /// touch the separately-projected K.
    pub parameter_free_v: bool,
    /// `None` = the op is absent, not a multiply by one.
    pub query_scale: Option<f32>,
    /// The canonical score-time multiply, kept separate from
    /// `query_scale` because folding them is algebra-equivalent and not
    /// fp-equivalent.
    pub score_scale: f32,
    pub position: LoweredPosition,
    /// Sliding window; `None` = attends the whole prefix.
    pub window: Option<usize>,
    /// `None` = the softcap op is absent.
    pub softcap: Option<f32>,
    /// The plan's residual-scale op (`LayerPlan::residual_scale`): the
    /// branch output (after any post-norm) is multiplied by this before
    /// its residual add. `None` = the op is absent, not a multiply by one.
    pub residual_scale: Option<f32>,
    /// Absolute position of the token being decoded.
    pub position_index: usize,
    /// Cache length **including** this position.
    pub kv_len: usize,
}

impl AttnShape {
    fn q_rows(&self) -> usize {
        self.num_q_heads * self.head_dim
    }
    fn kv_rows(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// Byte offset of this position's slot in a `[T, num_kv, head_dim]`
    /// cache.
    fn kv_slot_offset(&self) -> u64 {
        (self.position_index * self.kv_rows() * std::mem::size_of::<f32>()) as u64
    }
}

impl MetalBackend {
    /// Encode one attention op, hidden state in to hidden state out.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention(
        &self,
        encs: &mut dyn StageEncoders,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &AttnWeights<'_>,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
    ) {
        let (q_rows, kv_rows) = (shape.q_rows(), shape.kv_rows());
        let slot = shape.kv_slot_offset();

        // 1. pre-attention norm.
        let enc = encs.stage(Stage::AttnNorm);
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        // 2. projections. K and V land in their cache slots directly;
        //    a projection's bias joins its output there, before anything
        //    downstream reads it.
        let enc = encs.stage(Stage::AttnProj);
        let projections = [
            (&w.q, w.q_bias, s.q, 0u64, q_rows),
            (&w.k, w.k_bias, s.k_cache, slot, kv_rows),
            (&w.v, w.v_bias, s.v_cache, slot, kv_rows),
        ];
        // A-5b: three NVFP4 projections of one input are one dispatch —
        // the per-dispatch α paid once. Any other mix encodes per matrix.
        let fused: Option<Vec<Nvfp4Segment<'_>>> = projections
            .iter()
            .map(|(p, _, out, off, n)| nvfp4_segment(p, out, *off, *n))
            .collect();
        match fused {
            Some(segments) => {
                self.encode_nvfp4_matvec_segments(enc, s.normed, shape.hidden, &segments)
            }
            None => {
                for (p, _, out, off, n) in &projections {
                    self.encode_matvec(
                        enc,
                        p,
                        &MatvecTarget {
                            x: s.normed,
                            out,
                            out_offset: *off,
                            n: *n,
                            k: shape.hidden,
                        },
                    );
                }
            }
        }
        for (_, bias, out, off, n) in &projections {
            if let Some(bias) = bias {
                self.encode_bias_add(enc, out, *off, bias, *n);
            }
        }
        if let Some(g) = &w.gate {
            // The gate reads the *attention input* — the same normalised
            // vector the projections read, per `GateSource::AttentionInput`.
            self.encode_matvec(
                enc,
                g,
                &MatvecTarget {
                    x: s.normed,
                    out: s.gate,
                    out_offset: 0,
                    n: q_rows,
                    k: shape.hidden,
                },
            );
        }
        // 3. norms on the projections. V first: its norm reads the raw
        //    projection (on a K≡V layer V was projected from the K matrix
        //    into its own slot, so the key's norm below does not touch it).
        let enc = encs.stage(Stage::AttnQkOps);
        if shape.parameter_free_v {
            self.encode_parameter_free_qk_norm(
                enc,
                s.v_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if let Some(qk) = &w.qk_norm {
            self.encode_weighted_qk_norm(
                enc,
                s.q,
                0,
                qk.q,
                shape.num_q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
            self.encode_weighted_qk_norm(
                enc,
                s.k_cache,
                slot,
                qk.k,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
        }
        //    Parameter-free QK norm — Q and K independently, per head.
        if shape.parameter_free_q {
            self.encode_parameter_free_qk_norm(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if shape.parameter_free_k {
            self.encode_parameter_free_qk_norm(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        // 4. query scale — Q only, after the norm, before RoPE.
        if let Some(scale) = shape.query_scale {
            self.encode_scale_vector(enc, s.q, q_rows, scale);
        }
        // 5. position encoding, from this layer's policy. `None` means
        //    the op is absent and nothing is encoded — not rotation by
        //    a zero angle, which would also be a no-op here but would
        //    be the wrong reason.
        if let Some(amplitude) = shape.position.amplitude() {
            self.encode_rope(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
                amplitude,
            );
            self.encode_rope(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
                amplitude,
            );
        }
        // 6. attention over the cache.
        let enc = encs.stage(Stage::AttnCore);
        self.encode_kv_attention(enc, s, shape, w.sinks, 0, 0);
        // 7. the judged gate, then the output projection.
        let aggregated = match &w.gate {
            Some(_) => {
                let enc = encs.stage(Stage::AttnOut);
                self.encode_sigmoid_gate(enc, s.concat, s.gate, s.gated, q_rows);
                s.gated
            }
            None => s.concat,
        };
        // A-5b rung 2a: under two-norm placement with no output bias the
        // residual add folds into the o-proj write (bit-identical: the
        // same fp32 add), saving one dispatch per layer. Otherwise the
        // projection, bias and branch-norm/residual encode as before.
        let fused_out = match (w.post_norm.as_ref(), w.o_bias) {
            (None, None) if shape.residual_scale.is_none() && nvfp4_residual_fusion_enabled() => {
                nvfp4_segment(&w.o, h_out, 0, shape.hidden)
            }
            _ => None,
        };
        // The projection is its own stage so the profiler can price the
        // GEMV apart from the gate, bias, branch norm and residual around
        // it; production's `SingleEncoder` ignores the mark.
        let enc = encs.stage(Stage::AttnOProj);
        match fused_out {
            // `_sliced` carries the segment's byte offsets: under the
            // packed attention layout `o` is a row slice of the shared
            // allocation, and binding it at offset 0 would silently
            // compute the Q projection's rows instead.
            Some(seg) => self.encode_nvfp4_matvec_residual_sliced(
                enc,
                &MatvecOperands {
                    packed: seg.packed,
                    scales: seg.scales,
                    x: aggregated,
                    out: h_out,
                    out_offset: 0,
                    n: shape.hidden,
                    k: q_rows,
                },
                seg.tensor_scale,
                h_in,
                seg.packed_offset,
                seg.scales_offset,
            ),
            None => {
                self.encode_matvec(
                    enc,
                    &w.o,
                    &MatvecTarget {
                        x: aggregated,
                        out: s.attn_out,
                        out_offset: 0,
                        n: shape.hidden,
                        k: q_rows,
                    },
                );
                let enc = encs.stage(Stage::AttnOut);
                if let Some(bias) = w.o_bias {
                    self.encode_bias_add(enc, s.attn_out, 0, bias, shape.hidden);
                }
                // 8. post-attention norm (four-norm placement only), then
                //    the residual — branch output normalised *before* the add.
                self.encode_branch_norm_then_residual(
                    enc,
                    h_in,
                    s.attn_out,
                    h_out,
                    w.post_norm.as_ref(),
                    shape.hidden,
                    shape.residual_scale.unwrap_or(1.0),
                );
            }
        }
    }

    /// Attention over the cache: the same tiered dispatch the production
    /// decode path uses (`decode/encode_attn.rs`), transcribed rather than
    /// re-derived.
    ///
    /// - **Span tier.** `kv_attention` holds `SHORT_ATTENTION_SPAN` scores
    ///   in threadgroup memory; a span past it must run the long kernel,
    ///   whose bound is `LONG_ATTENTION_SPAN`. Before this the lowering
    ///   pinned the short kernel for every span, i.e. overflowed its
    ///   threadgroup scratch past 1024 — the sliding layers' 2048 window
    ///   and every full layer past 1K.
    /// - **Sequence-parallel phase 3 (KV-B1).** The weighted-V walk is
    ///   ~85% of long-span cost and the serial kernel walks it with
    ///   `head_dim` threads per head — 32 threadgroups regardless of
    ///   context on Glimmer. `ops::attention_geometry` resolves the
    ///   operator's `LARQL_KV_SEQPAR` against this op's full geometry
    ///   (head_dim, q heads, kv heads, span); serial for any geometry
    ///   without a measured row. The seqpar kernels bind the same
    ///   thirteen slots and differ only in a `slices x head_dim`
    ///   threadgroup and a reassociated sum, gated at 1e-4 against serial
    ///   in `tests/test_kernel_kv_attention_seqpar.rs`.
    fn encode_kv_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        q_offset: u64,
        out_offset: u64,
    ) {
        use crate::ops::kv_cache::{attention_span, LONG_ATTENTION_SPAN, SHORT_ATTENTION_SPAN};

        let window = shape.window.unwrap_or(0) as u32;
        let span = attention_span(shape.kv_len as u32, window);
        assert!(
            span as usize <= LONG_ATTENTION_SPAN,
            "attention span {span} exceeds the long kernel's threadgroup scratch \
             bound ({LONG_ATTENTION_SPAN})"
        );
        let long = span > SHORT_ATTENTION_SPAN;
        if self.encode_kv_attention_splitk(enc, s, shape, sinks, q_offset, out_offset, 1) {
            return;
        }
        // The planner, not the head_dim-only policy: this op's full
        // semantic geometry decides its execution geometry.
        let slices = crate::ops::attention_geometry::choose_attention_geometry(
            self.decode_flags.kv_seqpar,
            &crate::ops::attention_geometry::AttentionGeometryQuery {
                head_dim: shape.head_dim,
                num_q_heads: shape.num_q_heads,
                num_kv_heads: shape.num_kv_heads,
                span,
            },
        )
        .slices();
        let (pipeline, threads) = if slices > 1 {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SEQPAR);
            let p = if long {
                &self.attention.kv_attend_seqpar_long_pipeline
            } else {
                &self.attention.kv_attend_seqpar_pipeline
            };
            (p, (slices * shape.head_dim) as u64)
        } else {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SERIAL);
            let p = if long {
                &self.attention.kv_attend_long_pipeline
            } else {
                &self.attention.kv_attend_pipeline
            };
            (p, p.max_total_threads_per_threadgroup().min(256))
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.q), q_offset);
        enc.set_buffer(1, Some(s.k_cache), 0);
        enc.set_buffer(2, Some(s.v_cache), 0);
        enc.set_buffer(3, Some(s.concat), out_offset);
        super::set_u32(enc, 4, shape.kv_len as u32);
        super::set_u32(enc, 5, shape.head_dim as u32);
        super::set_u32(enc, 6, shape.num_q_heads as u32);
        super::set_u32(enc, 7, shape.num_kv_heads as u32);
        super::set_f32(enc, 8, shape.score_scale);
        super::set_u32(enc, 9, window);
        // Sinks: the plan's per-head logits when the layer carries the
        // judged semantics. The kernels read slot 10 only when `has_sinks`
        // is non-zero, but Metal still requires the binding to exist, so
        // a sink-free layer binds a one-float placeholder — `inv_freq` is
        // a live buffer of the right kind and is not read by this kernel.
        match sinks {
            Some(sinks) => {
                enc.set_buffer(10, Some(sinks), 0);
                super::set_u32(enc, 11, 1);
            }
            None => {
                enc.set_buffer(10, Some(s.inv_freq), 0);
                super::set_u32(enc, 11, 0);
            }
        }
        super::set_f32(enc, 12, shape.softcap.unwrap_or(0.0));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(shape.num_q_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// SPLITK-1 (`ops::kv_splitk`): when the planner has a split-K row
    /// for this geometry at every one of the `rows` positions' spans —
    /// and the SAME chunk/slice choice for all of them, so a verify row
    /// is the per-position dispatch's arithmetic exactly — encode both
    /// passes and return `true`. Otherwise encode nothing and return
    /// `false`, leaving the caller's seqpar/serial path.
    #[allow(clippy::too_many_arguments)]
    fn encode_kv_attention_splitk(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        q_offset: u64,
        out_offset: u64,
        rows: usize,
    ) -> bool {
        use crate::ops::attention_geometry::{choose_splitk, AttentionGeometryQuery};
        use crate::ops::kv_cache::attention_span;
        let Some(scratch) = s.splitk else {
            return false;
        };
        if !scratch.fits(rows, shape.num_q_heads, shape.q_rows()) {
            return false;
        }
        let window = shape.window.unwrap_or(0) as u32;
        let at = |i: usize| {
            choose_splitk(
                self.decode_flags.kv_seqpar,
                &AttentionGeometryQuery {
                    head_dim: shape.head_dim,
                    num_q_heads: shape.num_q_heads,
                    num_kv_heads: shape.num_kv_heads,
                    span: attention_span((shape.kv_len + i) as u32, window),
                },
            )
        };
        let Some(geometry) = at(0) else {
            return false;
        };
        if (1..rows).any(|i| at(i) != Some(geometry)) {
            return false;
        }
        crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SPLITK);
        crate::ops::kv_splitk::SplitKDispatch {
            q: s.q,
            q_offset,
            k_cache: s.k_cache,
            v_cache: s.v_cache,
            out: s.concat,
            out_offset,
            o_part: scratch.o_part,
            ml_part: scratch.ml_part,
            sinks,
            // A live buffer bound only to satisfy slot 6; never read
            // without sinks (as `encode_kv_attention`'s slot 10).
            placeholder: s.inv_freq,
            kv_len: shape.kv_len as u32,
            head_dim: shape.head_dim,
            num_q_heads: shape.num_q_heads,
            num_kv_heads: shape.num_kv_heads,
            score_scale: shape.score_scale,
            window,
            softcap: shape.softcap.unwrap_or(0.0),
            n_chunks: geometry.chunks,
            slices: geometry.slices,
            rows,
        }
        .encode(&self.attention, enc);
        true
    }
}

impl MetalBackend {
    /// VERIFY-N: one attention op over `rows` consecutive positions
    /// starting at `shape.position_index`, hidden states `[rows, hidden]`
    /// in and out.
    ///
    /// The same ordered program as [`Self::encode_attention`], with the
    /// projections batched — each weight stream read once for all rows —
    /// and everything position-dependent (norms, rotation, attention)
    /// dispatched per row at its own offset. Row `i` attends the cache up
    /// to and including position `base + i`, so the block is causal by
    /// construction: rows `i+1..` are written to the cache but never read
    /// by row `i`.
    ///
    /// `s`'s `normed`/`q`/`concat`/`attn_out` hold `rows` positions;
    /// `post_scratch` is `[rows, hidden]` and stands in for the post-norm's
    /// single-position scratch. Refuses the attention output gate, which
    /// no verified model needs yet.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_rows(
        &self,
        encs: &mut dyn StageEncoders,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &AttnWeights<'_>,
        s: &AttnScratch<'_>,
        post_scratch: &Buffer,
        shape: &AttnShape,
        rows: usize,
    ) -> Result<(), String> {
        if w.gate.is_some() {
            return Err("multi-position attention has no output-gate lowering yet".into());
        }
        let f = std::mem::size_of::<f32>() as u64;
        let (q_rows, kv_rows, hidden) = (shape.q_rows(), shape.kv_rows(), shape.hidden);
        let base = shape.position_index;
        let at = |i: usize| AttnShape {
            position_index: base + i,
            kv_len: base + i + 1,
            ..shape.clone()
        };
        let h_off = |i: usize| (i * hidden) as u64 * f;
        let q_off = |i: usize| (i * q_rows) as u64 * f;

        let enc = encs.stage(Stage::AttnNorm);
        self.encode_rms_norm_rows(
            enc,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            hidden,
            rows,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        let enc = encs.stage(Stage::AttnProj);
        let slot = shape.kv_slot_offset();
        for (m, out, off, n) in [
            (&w.q, s.q, 0u64, q_rows),
            (&w.k, s.k_cache, slot, kv_rows),
            (&w.v, s.v_cache, slot, kv_rows),
        ] {
            self.encode_matmul_rows(
                enc,
                m,
                &MatmulRowsTarget {
                    x: s.normed,
                    x_offset: 0,
                    out,
                    out_offset: off,
                    n,
                    k: hidden,
                    rows,
                },
            )?;
        }
        for i in 0..rows {
            let slot_i = at(i).kv_slot_offset();
            if let Some(b) = w.q_bias {
                self.encode_bias_add(enc, s.q, q_off(i), b, q_rows);
            }
            if let Some(b) = w.k_bias {
                self.encode_bias_add(enc, s.k_cache, slot_i, b, kv_rows);
            }
            if let Some(b) = w.v_bias {
                self.encode_bias_add(enc, s.v_cache, slot_i, b, kv_rows);
            }
        }
        // Per-head norms and rotation over the whole block: Q rows and the
        // block's K/V cache slots are each one contiguous `[rows, heads,
        // head_dim]` run, so a per-head kernel covers every position in
        // one dispatch with the same per-head arithmetic.
        let enc = encs.stage(Stage::AttnQkOps);
        let (q_heads, kv_heads) = (rows * shape.num_q_heads, rows * shape.num_kv_heads);
        if shape.parameter_free_v {
            self.encode_parameter_free_qk_norm(
                enc,
                s.v_cache,
                slot,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if let Some(qk) = &w.qk_norm {
            self.encode_weighted_qk_norm(
                enc,
                s.q,
                0,
                qk.q,
                q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
            self.encode_weighted_qk_norm(
                enc,
                s.k_cache,
                slot,
                qk.k,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
        }
        if shape.parameter_free_q {
            self.encode_parameter_free_qk_norm(
                enc,
                s.q,
                0,
                q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if shape.parameter_free_k {
            self.encode_parameter_free_qk_norm(
                enc,
                s.k_cache,
                slot,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        // Query scale is elementwise and position-free.
        if let Some(scale) = shape.query_scale {
            self.encode_scale_vector(enc, s.q, rows * q_rows, scale);
        }
        if let Some(amplitude) = shape.position.amplitude() {
            self.encode_rope_rows(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                s.inv_freq,
                base,
                amplitude,
                rows,
            );
            self.encode_rope_rows(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                s.inv_freq,
                base,
                amplitude,
                rows,
            );
        }
        let enc = encs.stage(Stage::AttnCore);
        if !self.encode_kv_attention_rows(enc, s, shape, w.sinks, rows) {
            for i in 0..rows {
                self.encode_kv_attention(enc, s, &at(i), w.sinks, q_off(i), q_off(i));
            }
        }
        let enc = encs.stage(Stage::AttnOut);
        self.encode_matmul_rows(
            enc,
            &w.o,
            &MatmulRowsTarget {
                x: s.concat,
                x_offset: 0,
                out: s.attn_out,
                out_offset: 0,
                n: hidden,
                k: q_rows,
                rows,
            },
        )?;
        if let Some(b) = w.o_bias {
            for i in 0..rows {
                self.encode_bias_add(enc, s.attn_out, h_off(i), b, hidden);
            }
        }
        self.encode_branch_norm_then_residual_rows(
            enc,
            h_in,
            s.attn_out,
            h_out,
            w.post_norm.as_ref(),
            post_scratch,
            hidden,
            rows,
            shape.residual_scale.unwrap_or(1.0),
        );
        Ok(())
    }
}

impl MetalBackend {
    /// VERIFY-N: the whole block's attention in ONE dispatch (grid
    /// `(num_q, rows)`), when every row would take the same short kernel —
    /// serial (`kv_attention_rows`) or sequence-parallel at one slice count
    /// (`kv_attention_seqpar_rows`). Each rows kernel shares its
    /// per-position kernel's body and threadgroup width, so every row is
    /// bit-identical to its per-position dispatch. Returns `false` (and
    /// encodes nothing) for a block per-position dispatch would route
    /// otherwise — a span past the short kernel, or rows that resolve to
    /// different geometries — so the caller keeps its loop for exactly
    /// those.
    fn encode_kv_attention_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        rows: usize,
    ) -> bool {
        use crate::ops::kv_cache::{attention_span, SHORT_ATTENTION_SPAN};
        // Split-K first: no short-span bound (each chunk holds its own
        // scores), same per-row arithmetic as the per-position dispatch.
        if self.encode_kv_attention_splitk(enc, s, shape, sinks, 0, 0, rows) {
            return true;
        }
        let window = shape.window.unwrap_or(0) as u32;
        // The last row has the longest span; every row must fit.
        let widest = attention_span((shape.kv_len + rows - 1) as u32, window);
        if widest > SHORT_ATTENTION_SPAN {
            return false;
        }
        // Every row must resolve to the SAME geometry — the per-position
        // path would dispatch each row with its own, and one dispatch can
        // only carry one.
        let slices_at = |i: usize| {
            let span = attention_span((shape.kv_len + i) as u32, window);
            crate::ops::attention_geometry::choose_attention_geometry(
                self.decode_flags.kv_seqpar,
                &crate::ops::attention_geometry::AttentionGeometryQuery {
                    head_dim: shape.head_dim,
                    num_q_heads: shape.num_q_heads,
                    num_kv_heads: shape.num_kv_heads,
                    span,
                },
            )
            .slices()
        };
        let slices = slices_at(0);
        if (1..rows).any(|i| slices_at(i) != slices) {
            return false;
        }
        // Each arm's threadgroup width is the per-position path's own, so
        // the reductions are identical row for row.
        let (pipeline, threads) = if slices > 1 {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SEQPAR);
            (
                &self.attention.kv_attend_seqpar_rows_pipeline,
                (slices * shape.head_dim) as u64,
            )
        } else {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SERIAL);
            (
                &self.attention.kv_attend_rows_pipeline,
                self.attention
                    .kv_attend_pipeline
                    .max_total_threads_per_threadgroup()
                    .min(256),
            )
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.q), 0);
        enc.set_buffer(1, Some(s.k_cache), 0);
        enc.set_buffer(2, Some(s.v_cache), 0);
        enc.set_buffer(3, Some(s.concat), 0);
        super::set_u32(enc, 4, shape.kv_len as u32);
        super::set_u32(enc, 5, shape.head_dim as u32);
        super::set_u32(enc, 6, shape.num_q_heads as u32);
        super::set_u32(enc, 7, shape.num_kv_heads as u32);
        super::set_f32(enc, 8, shape.score_scale);
        super::set_u32(enc, 9, window);
        match sinks {
            Some(sinks) => {
                enc.set_buffer(10, Some(sinks), 0);
                super::set_u32(enc, 11, 1);
            }
            None => {
                enc.set_buffer(10, Some(s.inv_freq), 0);
                super::set_u32(enc, 11, 0);
            }
        }
        super::set_f32(enc, 12, shape.softcap.unwrap_or(0.0));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(shape.num_q_heads as u64, rows as u64, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        true
    }
}
