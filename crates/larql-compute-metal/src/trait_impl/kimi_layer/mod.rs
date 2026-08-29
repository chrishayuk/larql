//! One complete Kimi decoder layer in ONE GPU ownership interval.
//!
//! Rung 5d. Rung 5c closed the attention; this closes the layer, and the
//! part that decides whether it IS closed is not the residuals or the
//! norms — it is whether Metal can compute the routing decision and
//! consume it in the grouped MoE **without the host ever seeing a
//! selected expert id**.
//!
//! ```text
//! hidden ─┬──────────────────────────────────────────┐  residual
//!         ↓                                          │
//!   input RMSNorm                                    │
//!         ↓                                          │
//!   KDA attention (device-resident state)            │
//!         ↓                                          ↓
//!   after_attention = hidden + attn ─────────────────┘─┐  residual
//!         ↓                                            │
//!   post-attention RMSNorm                             │
//!         ↓                                            │
//!   router: logits → sigmoid → +bias → top-k → renorm  │
//!         ↓  (GPU-written offset table AND weights)    │
//!   grouped MoE: top-k routed + shared                 │
//!         ↓                                            ↓
//!   layer output = after_attention + Σ w·expert ───────┘
//! ```
//!
//! One command buffer. One host→device upload (the layer input), one
//! device→host read (the layer output). Everything between — including
//! which experts ran — stays on device.
//!
//! **Residency is checked, never guessed.** The router writes offsets
//! out of a caller-supplied table mapping each expert to where it lives
//! in the resident bank; a selection of a non-resident expert is counted
//! on device and refused by the host after the wait. Which experts are
//! resident is the next problem, not this one's — what matters here is
//! that reading the wrong expert's weights is impossible rather than
//! merely unlikely.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::grouped_experts::GroupedError;
use super::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use super::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use crate::shaders::kimi_layer as layer_shader;
use crate::MetalBackend;

/// `rms_norm` must be dispatched as ONE threadgroup, this wide.
const NORM_THREADS_PER_TG: u64 = 256;
/// Plain `KimiRMSNorm` has no weight offset.
const NORM_WEIGHT_OFFSET: f32 = 0.0;
/// `residual_add`'s scale — the residual is unit here.
const RESIDUAL_UNIT_SCALE: f32 = 1.0;

/// The MoE half of a layer: the router, and the resident expert bank.
#[derive(Clone, Copy)]
pub struct KimiMoeWeights<'a> {
    /// `[experts, hidden]` f32.
    pub router_weight: &'a [f32],
    /// `[experts]` f32 — the correction bias. It SELECTS and never
    /// weighs; the weights come from the unbiased sigmoid scores.
    pub router_bias: &'a [f32],
    /// The three projection banks. Each carries its OWN addressing and
    /// its own shared-branch region, because a logical expert may sit
    /// at a different physical slot in each of them — and the shared
    /// branch may live in a different allocation entirely.
    pub gate: ProjectionBank<'a>,
    pub up: ProjectionBank<'a>,
    pub down: ProjectionBank<'a>,
    pub inter: usize,
    pub top_k: usize,
    pub renormalize: bool,
    /// `routed_scaling_factor`, folded into the routed weights.
    pub branch_scale: f32,
}

/// One physical region a grouped kernel can read: bytes, and what they
/// ARE.
///
/// Per projection, so `gate/up` at Q6_K with `down` at BF16 is
/// expressible — the precision map can already say it, so the physical
/// vocabulary must be able to carry it.
#[derive(Clone, Copy)]
pub struct EncodedRegion<'a> {
    pub bytes: &'a [u8],
    pub encoding: ExpertEncoding,
}

/// One projection's routed bank, how a logical expert is located in it,
/// and — separately — the shared branch's own region.
///
/// Bank and addressing travel together because they are one physical
/// fact. Splitting them let a caller pair a bank with another
/// projection's coordinates, which is the shared-coordinate bug this
/// whole change exists to make unrepresentable.
///
/// The shared branch is its OWN region, not an offset into the routed
/// bank: `Shared` vs `Routed` is semantic identity and must not imply
/// physical co-location. A source container keeps the shared expert in
/// the decoder stack and the routed experts in an expert bank; a
/// candidate overlay may compile the routed experts to Q6_K while the
/// shared branch stays source BF16. Co-locating the two is a layout an
/// artifact MAY choose (the region can be a subrange of the same
/// allocation), never an invariant execution relies on.
#[derive(Clone, Copy)]
pub struct ProjectionBank<'a> {
    pub routed: EncodedRegion<'a>,
    pub addressing: ExpertAddressing<'a>,
    /// The shared branch's `[n, k]` matrix for this projection, when
    /// the architecture has one. All three projections must agree on
    /// whether it exists — that is one semantic fact — and each binds
    /// its own bytes under its own encoding.
    pub shared: Option<EncodedRegion<'a>>,
}

/// A representation a grouped kernel can execute.
///
/// The backend answers whether it CAN run one; choosing it belongs to a
/// precision map. All three share one binding ABI — weights, byte
/// offsets, X, output, N/K, stride — so this selects a kernel and
/// nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertEncoding {
    Bf16,
    Q6K,
    Q4K,
}

impl ExpertEncoding {
    pub fn name(self) -> &'static str {
        match self {
            ExpertEncoding::Bf16 => "BF16",
            ExpertEncoding::Q6K => "Q6_K",
            ExpertEncoding::Q4K => "Q4_K",
        }
    }

    /// Bytes one `[n, k]` matrix occupies. `None` when the shape cannot
    /// be encoded this way at all.
    pub fn matrix_bytes(self, n: usize, k: usize) -> Option<usize> {
        match self {
            ExpertEncoding::Bf16 => Some(n * k * 2),
            ExpertEncoding::Q6K | ExpertEncoding::Q4K => k.is_multiple_of(256).then(|| {
                let per = if self == ExpertEncoding::Q6K {
                    210
                } else {
                    144
                };
                n * k / 256 * per
            }),
        }
    }
}

/// How the kernel turns a selected expert id into a bank byte offset.
///
/// **Addressability, not residency.** A full execution-shaped bank
/// addresses by identity and can never fail to answer; a packed subset
/// needs a table and genuinely can. Whether an addressable expert's
/// pages happen to be resident is a paging concern this type does not
/// express — manufacturing a 256-entry table for a full bank would state
/// a residency claim nobody made.
#[derive(Clone, Copy)]
pub enum ExpertAddressing<'a> {
    /// `offset = expert_id * stride`. Nothing is tabulated.
    ///
    /// One stride for all three projections, which holds because gate/up
    /// and down are transposes of one another — the same equality the
    /// shared offset table already relies on.
    Identity { experts: usize, stride: u32 },
    /// `[experts]` byte offsets, or [`layer_shader::NOT_RESIDENT`].
    Table(&'a [u32]),
}

impl ExpertAddressing<'_> {
    /// How many of the checkpoint's experts the router scores.
    pub fn experts(&self) -> usize {
        match self {
            ExpertAddressing::Identity { experts, .. } => *experts,
            ExpertAddressing::Table(t) => t.len(),
        }
    }

    /// The stride the kernel multiplies by, or 0 when it must consult
    /// the table instead.
    fn identity_stride(&self) -> u32 {
        match self {
            ExpertAddressing::Identity { stride, .. } => *stride,
            ExpertAddressing::Table(_) => 0,
        }
    }

    /// The offset an expert resolves to, host-side.
    pub fn offset_of(&self, expert: usize) -> Option<u32> {
        match self {
            ExpertAddressing::Identity { experts, stride } => {
                (expert < *experts).then(|| expert as u32 * stride)
            }
            ExpertAddressing::Table(t) => t
                .get(expert)
                .copied()
                .filter(|o| *o != layer_shader::NOT_RESIDENT),
        }
    }
}

/// Which attention operator a layer runs, with its weights, geometry and
/// resident state.
///
/// Kimi alternates KDA and full-attention layers, and R6c's whole point
/// is that a `KDA -> MLA -> KDA` run can share one command buffer — so
/// the attention is a parameter of the layer rather than a second layer
/// type. Everything after it (residual, norm, router, MoE, residual) is
/// identical either way and is written once.
#[derive(Clone, Copy)]
pub enum AttentionSpec<'a> {
    Kda {
        weights: KdaDeviceWeights<'a>,
        shape: KdaShape,
        state: &'a KdaDeviceState,
    },
    Mla {
        weights: MlaDeviceWeights<'a>,
        shape: MlaShape,
        state: &'a MlaDeviceState,
    },
}

impl AttentionSpec<'_> {
    pub fn hidden(&self) -> usize {
        match self {
            Self::Kda { shape, .. } => shape.hidden,
            Self::Mla { shape, .. } => shape.hidden,
        }
    }
}

/// One layer's weights, attention and MoE together.
#[derive(Clone, Copy)]
pub struct KimiLayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub post_attention_norm: &'a [f32],
    pub attention: AttentionSpec<'a>,
    pub ffn: ffn::FfnSpec<'a>,
    pub norm_eps: f32,
}

thread_local! {
    static ENCODE_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static WAIT_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// `(encode, wait)` milliseconds accumulated on this thread since the
/// last call, and reset.
///
/// A diagnostic split, not a production reading: the two have different
/// fixes, and a layer chain's host cost is not obviously one or the
/// other until it is measured.
pub fn take_chain_timing_ms() -> (f64, f64) {
    let e = ENCODE_MS.with(|c| c.replace(0.0));
    let w = WAIT_MS.with(|c| c.replace(0.0));
    (e, w)
}

/// Optional instrumentation for a chain, `None` in serving.
///
/// The router's decisions already exist on device in each layer's
/// `chosen` buffer — 8 `u32` a layer, ~832 bytes for a whole Kimi token.
/// Collecting them costs no extra command buffer, no extra dispatch and
/// **no per-layer synchronisation**: they are read after the chain's
/// single `commit`+`wait`, before the scratch is recycled.
///
/// That is deliberately not the traced path, which reads twelve
/// full-width planes a layer and cost 64 ms a token against 20 ms of
/// GPU work. Reading a handful of 32-byte buffers that are already
/// mapped is free by comparison — the expensive thing was always the
/// width of the planes, never the act of reading after the wait.
///
/// Ids come back in ROUTER ORDER: the raw execution fact. Deciding that
/// ordering is irrelevant belongs to the quality metric, not the engine.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExecutionTrace {
    /// One entry per layer in the chain, each `top_k` expert ids. Empty
    /// for a dense layer, which routes nothing.
    pub routes: Vec<Vec<u32>>,
}

/// One layer in a chain. The state travels inside
/// [`KimiLayerWeights::attention`], because which state a layer carries
/// is decided by which attention it runs.
#[derive(Clone, Copy)]
pub struct KimiLayerCall<'a> {
    pub weights: KimiLayerWeights<'a>,
}

/// One layer's attention scratch, whichever operator it runs.
enum AttentionScratch {
    Kda(crate::trait_impl::kda::Scratch),
    Mla(crate::trait_impl::mla::MlaScratch),
}

impl AttentionScratch {
    /// The attention output — the residual's second operand, and the
    /// only plane the layer path itself reads.
    fn out(&self) -> &Buffer {
        match self {
            Self::Kda(s) => &s.out,
            Self::Mla(s) => &s.out,
        }
    }
}

/// Scratch for one layer, one pop from the pool each so none alias.
pub(crate) struct LayerScratch {
    input_normed: Buffer,
    after_attention: Buffer,
    post_normed: Buffer,
    logits: Buffer,
    scores: Buffer,
    sel_scores: Buffer,
    chosen: Buffer,
    gate_offsets: Buffer,
    up_offsets: Buffer,
    down_offsets: Buffer,
    weights: Buffer,
    refusals: Buffer,
    gate_out: Buffer,
    up_out: Buffer,
    h: Buffer,
    expert_out: Buffer,
    out: Buffer,
}

impl MetalBackend {
    /// One decoder layer, one command buffer.
    pub fn kimi_decoder_layer(
        &self,
        w: KimiLayerWeights<'_>,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let p = self.kimi_decoder_layer_traced(w, x)?;
        Ok((p.output, p.gpu_ms))
    }

    /// **Several consecutive decoder layers in ONE command buffer.**
    ///
    /// Rung 5e. Layer `i+1`'s input is layer `i`'s output BUFFER — the
    /// hidden state never leaves the device, so layer `i+1`'s router
    /// scores a vector the host has not seen and its expert selection is
    /// dynamically downstream of layer `i`. That is the difference
    /// between chaining layers and merely batching them: a path that
    /// re-used the original input, or read a stale buffer, routes
    /// somewhere else and the gate catches it.
    ///
    /// Each layer carries its own recurrent state; `states` and
    /// `layers` are paired by position.
    pub fn kimi_decoder_layers(
        &self,
        layers: &[KimiLayerCall<'_>],
        x: &[f32],
        trace: Option<&mut ExecutionTrace>,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let hidden = layers
            .first()
            .ok_or(GroupedError::NoExpertsSelected)?
            .weights
            .attention
            .hidden();
        let (scratch, kda, _, gpu_ms) = self.encode_layer_chain(layers, None, x)?;
        collect_routes(layers, &scratch, trace);
        // ONE readback: the last layer's output. Everything else stays
        // on device. Reading the traced planes here instead cost 64 ms a
        // token in the real trajectory — 12 buffers per layer over 19
        // layers, against 20 ms of actual GPU work.
        let out = crate::buffers::read_buffer_f32(&scratch[scratch.len() - 1].out, hidden);
        self.recycle_chain(scratch, kda);
        Ok((out, gpu_ms))
    }

    fn recycle_chain(&self, scratch: Vec<LayerScratch>, kda_scratch: Vec<AttentionScratch>) {
        for s in scratch {
            self.recycle_layer(s);
        }
        for s in kda_scratch {
            match s {
                AttentionScratch::Kda(s) => self.recycle_scratch(s),
                AttentionScratch::Mla(s) => self.recycle_mla(s),
            }
        }
    }

    /// Encode the whole chain, commit, wait, and check every layer's
    /// refusal counter. Returns the scratch so the caller decides what
    /// to read back.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::type_complexity)]
    fn encode_layer_chain(
        &self,
        layers: &[KimiLayerCall<'_>],
        head: Option<&head::KimiHead<'_>>,
        x: &[f32],
    ) -> Result<
        (
            Vec<LayerScratch>,
            Vec<AttentionScratch>,
            Option<head::HeadScratch>,
            f64,
        ),
        GroupedError,
    > {
        if layers.is_empty() {
            return Err(GroupedError::NoExpertsSelected);
        }
        let hidden = layers[0].weights.attention.hidden();
        // Validate EVERY layer before encoding anything: an encoder
        // dropped without `end_encoding` aborts the process, so a
        // refusal found halfway through would not be recoverable. The
        // attention half validates through its own operator's checks.
        let mut visible = Vec::with_capacity(layers.len());
        for call in layers {
            let experts = call.weights.ffn.experts();
            visible.push(match call.weights.attention {
                AttentionSpec::Kda {
                    weights,
                    shape,
                    state,
                } => {
                    Self::validate_kda(weights, shape, state, hidden)?;
                    0
                }
                AttentionSpec::Mla { shape, state, .. } => {
                    Self::validate_mla(shape, state, hidden)?
                }
            });
            validate_layer(&call.weights, experts, call.weights.ffn.slots(), hidden)?;
        }
        if let Some(h) = head {
            Self::validate_head(h, hidden)?;
        }
        if x.len() != hidden {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: hidden,
                have: x.len(),
            });
        }

        let mut kda_scratch = Vec::with_capacity(layers.len());
        let mut scratch = Vec::with_capacity(layers.len());
        for (call, &vis) in layers.iter().zip(&visible) {
            kda_scratch.push(match call.weights.attention {
                AttentionSpec::Kda { shape, .. } => AttentionScratch::Kda(self.kda_scratch(shape)),
                AttentionSpec::Mla { shape, .. } => {
                    AttentionScratch::Mla(self.mla_scratch(shape, vis))
                }
            });
            scratch.push(self.layer_scratch(
                hidden,
                call.weights.ffn.inter(),
                call.weights.ffn.experts(),
                call.weights.ffn.slots(),
                call.weights.ffn.top_k(),
            ));
        }
        let buf_x = self.bufs().transient_from_f32(x);

        let encode_clock = std::time::Instant::now();
        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // Buffers bound into the encoder must outlive the wait; the
        // cache clones held here guarantee it.
        let mut held = Vec::with_capacity(layers.len());
        for (i, call) in layers.iter().enumerate() {
            // The chain: layer 0 reads the upload, every later layer
            // reads its predecessor's OUTPUT buffer. Nothing in between
            // touches the host.
            let input = if i == 0 { &buf_x } else { &scratch[i - 1].out };
            held.push(self.encode_kimi_layer(
                enc,
                call.weights,
                input,
                &kda_scratch[i],
                &scratch[i],
                visible[i],
            ));
        }
        // The head reads the LAST layer's output buffer, inside this
        // same encoder — so the hidden state never crosses to the host
        // and the token stays at one epoch.
        let head_scratch = head.map(|h| {
            let s = self.kimi_head_scratch(hidden, h.vocab);
            held.push(self.encode_kimi_head(enc, h, &scratch[layers.len() - 1].out, &s, hidden));
            s
        });
        enc.end_encoding();
        let encode_ms = encode_clock.elapsed().as_secs_f64() * 1000.0;
        let wait_clock = std::time::Instant::now();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/kimi_layer/mod.rs:layers",
        );
        let wait_ms = wait_clock.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        // Encode-vs-wait, because they have different fixes: encode is
        // host work per bound resource, wait is submission latency plus
        // GPU execution. Rung 5f's fixture bound 40 MiB a layer and this
        // binds ~900 MB, so the two do not scale together.
        ENCODE_MS.with(|c| c.set(c.get() + encode_ms));
        WAIT_MS.with(|c| c.set(c.get() + wait_ms));
        drop(held);

        // Each MLA layer's cache advanced by one position — but only
        // once the dispatch that wrote the latent has actually run, so
        // this happens after the wait and not at encode time.
        for (call, &vis) in layers.iter().zip(&visible) {
            if let AttentionSpec::Mla { state, .. } = call.weights.attention {
                state.advance_to(vis);
            }
        }

        // Every layer's refusal counter, read AFTER the wait. A route
        // that named a non-resident expert produced numbers from slot
        // 0's weights, and returning them would be the silent wrong
        // answer this seam exists to prevent.
        let mut refusal: Option<(usize, u32)> = None;
        for (i, s) in scratch.iter().enumerate() {
            let n = read_u32(&s.refusals, 1)[0];
            if n != 0 && refusal.is_none() {
                refusal = Some((i, n));
            }
        }
        if let Some((layer, count)) = refusal {
            self.recycle_chain(scratch, kda_scratch);
            return Err(GroupedError::LayerRouteNotResident {
                layer,
                refusals: count,
            });
        }
        Ok((scratch, kda_scratch, head_scratch, gpu_ms))
    }

    /// Encode one layer into an existing encoder, reading `input` and
    /// writing `s.out`. Returns the device buffers it bound, which the
    /// caller must hold until the wait.
    #[allow(clippy::too_many_arguments)]
    fn encode_kimi_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: KimiLayerWeights<'_>,
        input: &Buffer,
        attention: &AttentionScratch,
        s: &LayerScratch,
        visible: usize,
    ) -> Vec<Buffer> {
        let hidden = w.attention.hidden();
        let f32b = |v: &[f32]| self.bufs().get_f32(v);
        let (norm_in, norm_post) = (f32b(w.input_norm), f32b(w.post_attention_norm));

        self.encode_rms_norm(enc, input, &norm_in, &s.input_normed, hidden, w.norm_eps);
        match (w.attention, attention) {
            (
                AttentionSpec::Kda {
                    weights,
                    shape,
                    state,
                },
                AttentionScratch::Kda(k),
            ) => self.encode_kda_attention(enc, weights, shape, state, &s.input_normed, k),
            (
                AttentionSpec::Mla {
                    weights,
                    shape,
                    state,
                },
                AttentionScratch::Mla(m),
            ) => self.encode_mla_attention_into(
                enc,
                weights,
                shape,
                state,
                &s.input_normed,
                m,
                visible,
            ),
            _ => unreachable!("the scratch is built from the same spec"),
        }
        // `after_attention = input + attention`. The attention plane
        // itself is `kda_scratch.out`, so nothing needs copying.
        self.encode_residual_add(
            enc,
            input,
            attention.out(),
            &s.after_attention,
            hidden,
            RESIDUAL_UNIT_SCALE,
        );
        self.encode_rms_norm(
            enc,
            &s.after_attention,
            &norm_post,
            &s.post_normed,
            hidden,
            w.norm_eps,
        );

        let ffn_held = match &w.ffn {
            ffn::FfnSpec::Moe(m) => self.encode_moe_ffn(enc, m, s, hidden),
            ffn::FfnSpec::Dense(d) => self.encode_dense_ffn(enc, d, s, hidden),
        };

        let mut held = vec![norm_in, norm_post];
        held.extend(ffn_held);
        held
    }

    fn layer_scratch(
        &self,
        hidden: usize,
        inter: usize,
        experts: usize,
        slots: usize,
        top_k: usize,
    ) -> LayerScratch {
        // A dense layer has no router, so `experts` and `top_k` are
        // zero and the router planes below are never bound. Metal
        // refuses a zero-length buffer, so allocate one element rather
        // than special-case the struct: four bytes, never read.
        let f = |n: usize| self.bufs().output((n.max(1) * 4) as u64);
        let refusals = f(1);
        // Zeroed from the host before the encoder opens: the pool
        // recycles, and a recycled counter carries the previous route's
        // refusals.
        let ptr = refusals.contents() as *mut u32;
        if !ptr.is_null() {
            // SAFETY: a pooled 4-byte buffer not yet bound to any
            // encoder, so the GPU is not reading it.
            unsafe { std::ptr::write(ptr, 0) };
        }
        LayerScratch {
            input_normed: f(hidden),
            after_attention: f(hidden),
            post_normed: f(hidden),
            logits: f(experts),
            scores: f(experts),
            sel_scores: f(experts),
            chosen: f(top_k),
            // Routed slots only: the shared branch owns no entry in the
            // address tables — its region is bound directly.
            gate_offsets: f(top_k),
            up_offsets: f(top_k),
            down_offsets: f(top_k),
            // Always `top_k + 1`: the router writes the shared branch's
            // constant 1.0 unconditionally, and a layer without a shared
            // branch simply never reads it.
            weights: f(top_k + 1),
            refusals,
            gate_out: f(slots * inter),
            up_out: f(slots * inter),
            h: f(slots * inter),
            expert_out: f(slots * hidden),
            out: f(hidden),
        }
    }

    fn recycle_layer(&self, s: LayerScratch) {
        for b in [
            s.input_normed,
            s.after_attention,
            s.post_normed,
            s.logits,
            s.scores,
            s.sel_scores,
            s.chosen,
            s.gate_offsets,
            s.up_offsets,
            s.down_offsets,
            s.weights,
            s.refusals,
            s.gate_out,
            s.up_out,
            s.h,
            s.expert_out,
            s.out,
        ] {
            self.bufs().recycle(b);
        }
    }

    fn encode_rms_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        len: usize,
        eps: f32,
    ) {
        let n = len as u32;
        enc.set_compute_pipeline_state(&self.norms.rms_norm_pipeline);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            5,
            4,
            &NORM_WEIGHT_OFFSET as *const f32 as *const std::ffi::c_void,
        );
        // One threadgroup, per the kernel's own contract.
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(NORM_THREADS_PER_TG, 1, 1),
        );
    }

    fn encode_router_select(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &KimiMoeWeights<'_>,
        router_bias: &Buffer,
        s: &LayerScratch,
        experts: usize,
    ) {
        let (e, k) = (experts as u32, moe.top_k as u32);
        let renorm: u32 = u32::from(moe.renormalize);
        enc.set_compute_pipeline_state(&self.kimi.router_select);
        enc.set_buffer(0, Some(&s.logits), 0);
        enc.set_buffer(1, Some(router_bias), 0);
        enc.set_buffer(2, Some(&s.scores), 0);
        enc.set_buffer(3, Some(&s.sel_scores), 0);
        enc.set_buffer(4, Some(&s.chosen), 0);
        enc.set_buffer(5, Some(&s.weights), 0);
        enc.set_bytes(6, 4, &e as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &renorm as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(
            9,
            4,
            &moe.branch_scale as *const f32 as *const std::ffi::c_void,
        );
        // One threadgroup: the selection is serial by design.
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(layer_shader::SELECT_THREADS_PER_TG, 1, 1),
        );
    }

    /// `out = residual + sum_slot weight[slot] * branch[slot]`.
    ///
    /// The combine weights come from a DEVICE buffer, so the routed path
    /// passes the router's own output and the dense path passes a
    /// constant one — the same kernel, and the dense layer's residual is
    /// therefore the residual the routed layers already prove.
    pub(crate) fn encode_moe_combine(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &LayerScratch,
        weights: &Buffer,
        hidden: usize,
        slots: usize,
    ) {
        let (h, k) = (hidden as u32, slots as u32);
        enc.set_compute_pipeline_state(&self.kimi.moe_combine);
        enc.set_buffer(0, Some(&s.expert_out), 0);
        enc.set_buffer(1, Some(&s.after_attention), 0);
        enc.set_buffer(2, Some(weights), 0);
        enc.set_buffer(3, Some(&s.out), 0);
        enc.set_bytes(4, 4, &h as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k as *const u32 as *const std::ffi::c_void);
        crate::lowering::dispatch_linear(enc, &self.kimi.moe_combine, hidden);
    }
}

/// Host-side checks the router's own kernel cannot make: the sizes it is
/// about to index, and that every RESIDENT expert really lies inside the
/// banks. A non-resident selection is the kernel's business; an
/// in-bounds-looking offset that is not is this function's.
fn validate_layer(
    w: &KimiLayerWeights<'_>,
    experts: usize,
    slots: usize,
    hidden: usize,
) -> Result<(), GroupedError> {
    let moe = match &w.ffn {
        ffn::FfnSpec::Moe(m) => m,
        ffn::FfnSpec::Dense(d) => return ffn::FfnSpec::validate_dense(d, hidden),
    };
    if experts == 0 || moe.top_k == 0 {
        return Err(GroupedError::NoExpertsSelected);
    }
    if experts > layer_shader::MAX_EXPERTS || slots > layer_shader::MAX_SLOTS {
        return Err(GroupedError::SlotCountMismatch {
            expected: layer_shader::MAX_EXPERTS.min(layer_shader::MAX_SLOTS),
            found: experts.max(slots),
        });
    }
    // Whether a shared expert exists is ONE semantic fact; the three
    // projections declaring it differently would silently drop one
    // projection's shared contribution.
    let has_shared = moe.gate.shared.is_some();
    for bank in [&moe.up, &moe.down] {
        if bank.shared.is_some() != has_shared {
            return Err(GroupedError::SharedBranchInconsistent);
        }
    }
    for bank in [&moe.gate, &moe.up, &moe.down] {
        if bank.addressing.experts() != experts {
            return Err(GroupedError::SlotCountMismatch {
                expected: experts,
                found: bank.addressing.experts(),
            });
        }
    }
    if moe.router_weight.len() != experts * hidden {
        return Err(GroupedError::SlotCountMismatch {
            expected: experts * hidden,
            found: moe.router_weight.len(),
        });
    }
    for (name, bank, n, k) in [
        (0usize, moe.gate, moe.inter, hidden),
        (1, moe.up, moe.inter, hidden),
        (2, moe.down, hidden, moe.inter),
    ] {
        // Sized at what the bank CLAIMS to be, not at bf16. Bytes that
        // are Q6_K dispatched as BF16 need more room than they have and
        // are refused here; the opposite direction is caught where the
        // bank's exact extent is known, since a shifted view over a
        // whole segment legitimately has room to spare.
        let per = bank
            .routed
            .encoding
            .matrix_bytes(n, k)
            .ok_or(GroupedError::KNotSuperblockAligned { k })?;
        // Every ADDRESSABLE expert must lie inside the routed bank. For
        // an identity bank that is every expert; for a packed one only
        // those the table names.
        for off in (0..experts).filter_map(|e| bank.addressing.offset_of(e)) {
            let need = off as usize + per;
            if need > bank.routed.bytes.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot: name,
                    offset: off,
                    need,
                    have: bank.routed.bytes.len(),
                });
            }
        }
        // The shared branch's own region, under its OWN encoding —
        // which need not be the routed bank's.
        if let Some(shared) = &bank.shared {
            let need = shared
                .encoding
                .matrix_bytes(n, k)
                .ok_or(GroupedError::KNotSuperblockAligned { k })?;
            if need > shared.bytes.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot: name,
                    offset: 0,
                    need,
                    have: shared.bytes.len(),
                });
            }
        }
    }
    Ok(())
}

/// Read each layer's selected expert ids, after the wait and before the
/// scratch is recycled.
fn collect_routes(
    layers: &[KimiLayerCall<'_>],
    scratch: &[LayerScratch],
    trace: Option<&mut ExecutionTrace>,
) {
    let Some(trace) = trace else {
        return;
    };
    trace.routes.clear();
    for (call, s) in layers.iter().zip(scratch) {
        let top_k = call.weights.ffn.top_k();
        // A dense layer routes nothing, and its `chosen` buffer is the
        // one-element placeholder `layer_scratch` allocates. Reading it
        // would report an expert that was never selected.
        trace.routes.push(if top_k == 0 {
            Vec::new()
        } else {
            read_u32(&s.chosen, top_k)
        });
    }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and no invalid bit patterns, and `u8`
    // has weaker alignment, so any `&[u32]` is a valid `&[u8]` of four
    // times the length for the same lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn read_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    if ptr.is_null() {
        return vec![0; n];
    }
    // SAFETY: shared-storage buffer of at least `n * 4` bytes, read after
    // `wait_until_completed`, so no GPU work is in flight against it.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

mod ffn;
mod head;
mod traced;
pub use ffn::{FfnSpec, KimiDenseFfn};
pub use head::KimiHead;
pub use traced::KimiLayerPlanes;

#[cfg(test)]
mod tests;
