//! The mixed stack: Metal KDA+MoE segments, CPU MLA layers.
//!
//! Rungs 5e/5f proved N chained KDA+MoE layers execute autonomously in
//! one command buffer. Kimi Linear's real topology is not N contiguous
//! KDA layers though — it alternates, with full-attention (MLA) layers
//! at 3, 7, 11, 15, 19, 23 and 26. Only KDA is ported, so the real
//! decode is:
//!
//! ```text
//! [ Metal: KDA layers 0..2 ]  CPU: MLA 3  [ Metal: 4..6 ]  CPU: MLA 7  ...
//! ```
//!
//! roughly **three KDA layers per GPU epoch**. Rung 5a's curve says four
//! blocks per command buffer capture ~88% of the epoch prize, so three
//! collects most of it — this topology is close to the measured sweet
//! spot by accident rather than design.
//!
//! **What crosses, and why that is the whole point.** A device run ends
//! at an MLA layer because MLA is not ported; its hidden state must come
//! back for the host to run attention, and go out again for the next
//! run. Those are deliberate epoch boundaries, priced at ~0.23-0.26 ms
//! each by rungs 5a/5f, and there are only seven of them a token instead
//! of twenty-six.
//!
//! **State lives where its layer runs.** A device layer's recurrent and
//! convolution state is a `KdaDeviceState` that never comes back; a host
//! layer's is the CPU `LayerState` the proven path already carries.
//! Neither is copied into the other, because a state that round-trips is
//! a crossing this module exists to avoid.

use crate::error::VindexError;
use crate::format::vindex3::represent::physical::{
    EncodedRegion, ExpertBankBinding, ExpertEncoding as PhysEncoding, ExpertLayout,
};
use larql_compute_metal::trait_impl::grouped_experts::{ExpertOffset, GroupedError};
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::{
    AttentionSpec, ExecutionTrace, ExpertAddressing, ExpertEncoding, FfnSpec, KimiDenseFfn,
    KimiHead, KimiLayerCall, KimiLayerWeights, KimiMoeWeights, ProjectionBank,
};
use larql_compute_metal::trait_impl::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use larql_compute_metal::MetalBackend;
use std::time::Instant;

use super::stack::{layer_forward_public, LayerSpec, LayerState, StackLayerTrace};

/// One layer's attention operands, prepared for the device.
///
/// Which variant a layer holds is the checkpoint's decision. Kimi
/// alternates KDA and full attention, and once BOTH are ported the only
/// host break left in a token is the dense layer 0 — so this enum is
/// what collapses seven GPU epochs into one.
pub enum DeviceAttn {
    Kda {
        /// `q|k|v` concatenated; the grouped kernel binds one buffer.
        qkv_bank: Vec<u8>,
        qkv_offsets: [ExpertOffset; 3],
        o_proj: Vec<u8>,
        /// conv1d x3, f_a, f_b, g_a, g_b, b_proj, a_log, dt_bias, o_norm
        /// — `KdaDeviceWeights`'s own field order.
        f32s: Vec<Vec<f32>>,
    },
    Mla {
        q: Vec<u8>,
        kv_a: Vec<u8>,
        kv_b: Vec<u8>,
        o: Vec<u8>,
        kv_a_norm: Vec<f32>,
    },
}

/// The resident state a layer carries — of the kind its attention needs,
/// never shared, never the wrong kind.
pub enum DeviceState {
    Kda(KdaDeviceState),
    Mla(MlaDeviceState),
}

impl DeviceState {
    /// Start a new sequence. Neither kind reallocates.
    pub fn reset(&self) {
        match self {
            Self::Kda(s) => s.reset(),
            Self::Mla(s) => s.reset(),
        }
    }
}

/// One KDA+MoE layer prepared for the device.
///
/// Everything the grouped kernels need in the layout they need it: the
/// three projections concatenated into one `q|k|v` bank, the resident
/// experts concatenated per projection, and a residency table mapping
/// expert id to a byte offset in those banks.
///
/// Owned rather than borrowed because the banks do not exist anywhere
/// else — the checkpoint stores per-expert tensors and the grouped
/// kernel binds one buffer. A container that stored the bank contiguously
/// would let this borrow instead.
pub struct DeviceLayer {
    pub attn: DeviceAttn,
    pub state: DeviceState,
    /// The expert bank as three physical regions plus the layout that
    /// addresses them.
    ///
    /// The layer does not know whether those regions are owned fixture
    /// bytes, an mmap'd source segment, a compiled candidate overlay or
    /// a future hot cache — that is the point of the binding sitting one
    /// level above it.
    pub bank: ExpertBankBinding,
    /// Byte offsets per expert for a `Mapped` bank. Empty for
    /// `Identity`, which tabulates nothing.
    pub offsets: Vec<u32>,
    /// Bytes one expert occupies in a gate/up bank — what `Identity`
    /// multiplies by.
    pub expert_stride: u32,
    pub shared_offset: u32,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub inter: usize,
    pub top_k: usize,
    /// This layer's three banks hold ONE gated MLP, not a bank of
    /// experts, and it has no router. Kimi's layer 0 is the only one.
    pub dense: bool,
    pub renormalize: bool,
    pub branch_scale: f32,
    pub norm_eps: f32,
    /// The geometries this layer's attention may run at; the enum picks.
    pub kda_shape: KdaShape,
    pub mla_shape: MlaShape,
    pub mla_norm_eps: f32,
}

impl DeviceLayer {
    /// The attention half's weight allocations, for residency
    /// registration. A ~17 GiB working set with implicit
    /// per-command-buffer residency cost 39 ms a token; these have to be
    /// declared like the expert banks.
    pub fn attention_banks(&self) -> Vec<&[u8]> {
        match &self.attn {
            DeviceAttn::Kda {
                qkv_bank, o_proj, ..
            } => vec![qkv_bank, o_proj],
            DeviceAttn::Mla {
                q, kv_a, kv_b, o, ..
            } => vec![q, kv_a, kv_b, o],
        }
    }

    /// The device weights, for a caller driving `kimi_decoder_layers`
    /// directly — a timing probe, or a future executor.
    pub fn weights_public(&self) -> KimiLayerWeights<'_> {
        self.weights()
    }

    /// The attention half, bound to this layer's own state. A mismatch
    /// between the two is a construction error, not a runtime fallback.
    fn attention(&self) -> AttentionSpec<'_> {
        match (&self.attn, &self.state) {
            (
                DeviceAttn::Kda {
                    qkv_bank,
                    qkv_offsets,
                    o_proj,
                    f32s: f,
                },
                DeviceState::Kda(state),
            ) => AttentionSpec::Kda {
                weights: KdaDeviceWeights {
                    qkv_bank,
                    qkv_offsets,
                    o_proj,
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
                    norm_eps: self.norm_eps,
                },
                shape: self.kda_shape,
                state,
            },
            (
                DeviceAttn::Mla {
                    q,
                    kv_a,
                    kv_b,
                    o,
                    kv_a_norm,
                },
                DeviceState::Mla(state),
            ) => AttentionSpec::Mla {
                weights: MlaDeviceWeights {
                    q_proj: q,
                    kv_a_proj: kv_a,
                    kv_a_norm,
                    kv_b_proj: kv_b,
                    o_proj: o,
                    kv_a_norm_eps: self.mla_norm_eps,
                },
                shape: self.mla_shape,
                state,
            },
            _ => panic!("a layer's state must match its attention kind"),
        }
    }

    /// Refuse a layer whose banks are not what they claim to be.
    ///
    /// Called at CONSTRUCTION, which is the last point where region
    /// extent, declared encoding and projection shape are all known
    /// together — after this the three travel separately and no one can
    /// compare them again. A bank whose bytes are BF16 while its
    /// encoding says Q6_K would otherwise dispatch the Q6 kernel over
    /// valid bf16 bytes and produce plausible output from the wrong
    /// representation, which is the worst failure available here.
    pub fn validate_banks(&self, hidden: usize) -> Result<(), VindexError> {
        self.bank.validate(hidden, self.inter)
    }

    /// One projection's bank paired with how a logical expert is
    /// located in it.
    fn projection<'a>(&'a self, r: &'a EncodedRegion) -> ProjectionBank<'a> {
        ProjectionBank {
            bytes: r.region.bytes(),
            addressing: match &self.bank.layout {
                ExpertLayout::Identity { experts } => ExpertAddressing::Identity {
                    experts: *experts as usize,
                    stride: self.expert_stride,
                },
                ExpertLayout::Mapped { .. } => ExpertAddressing::Table(&self.offsets),
            },
            shared_offset: self.shared_offset,
            // The REGION knows what its bytes are; the layer never
            // declares it, so a declaration cannot drift from the bytes.
            encoding: match r.encoding {
                PhysEncoding::Bf16 => ExpertEncoding::Bf16,
                PhysEncoding::Q6K => ExpertEncoding::Q6K,
                PhysEncoding::Q4K => ExpertEncoding::Q4K,
            },
        }
    }

    fn weights(&self) -> KimiLayerWeights<'_> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: self.attention(),
            ffn: if self.dense {
                debug_assert!(
                    self.router_bias.is_empty() && self.router_weight.is_empty(),
                    "a dense layer carries no router; a populated one here means \
                     the flag and the weights disagree"
                );
                FfnSpec::Dense(KimiDenseFfn {
                    gate: self.bank.gate.region.bytes(),
                    up: self.bank.up.region.bytes(),
                    down: self.bank.down.region.bytes(),
                    inter: self.inter,
                })
            } else {
                FfnSpec::Moe(KimiMoeWeights {
                    router_weight: &self.router_weight,
                    router_bias: &self.router_bias,
                    // Each projection carries its own addressing. Today
                    // the three share a layout, which is a fact about
                    // this bank rather than something the types assume:
                    // a compiled overlay may give one logical expert a
                    // different physical slot in each.
                    gate: self.projection(&self.bank.gate),
                    up: self.projection(&self.bank.up),
                    down: self.projection(&self.bank.down),
                    inter: self.inter,
                    top_k: self.top_k,
                    renormalize: self.renormalize,
                    branch_scale: self.branch_scale,
                })
            },
            norm_eps: self.norm_eps,
        }
    }
}

/// Where one token's time went.
///
/// Reported by stage from the start, per rung 5d's lesson: a single
/// serial GPU operation once cost more than the attention and the whole
/// MoE together and was invisible in the total.
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridTiming {
    /// Wall inside the device epochs, including submission and readback.
    pub device_wall_ms: f64,
    /// GPU-busy inside those epochs.
    pub device_gpu_ms: f64,
    /// Wall in the host layers.
    pub host_wall_ms: f64,
    pub epochs: usize,
}

/// One model layer: prepared for the device, or left on the host.
///
/// Two parallel arrays rather than one enum of both, because a device
/// RUN needs an immutable slice while a host layer needs its state
/// mutably, and those borrows have to be disjoint.
pub struct HybridStack<'a> {
    /// `Some` for every layer that runs on device.
    device: Vec<Option<DeviceLayer>>,
    /// `Some` for every layer that stays on the host, with its state.
    host: Vec<Option<(LayerSpec<'a>, LayerState)>>,
    /// The model head, when it runs on device. See [`Self::attach_head`].
    head: Option<HybridHead>,
}

/// Final norm + vocabulary projection, held for the device.
///
/// `weight` is `vocab x hidden` row-major bf16 — half the bytes of the
/// f32 the fixture stores, and for a checkpoint that was bf16 to begin
/// with, the same values exactly.
pub struct HybridHead {
    pub norm_weight: Vec<f32>,
    pub weight: Vec<u8>,
    pub vocab: usize,
    pub norm_eps: f32,
}

impl<'a> HybridStack<'a> {
    /// `slots` is one entry per model layer, in layer order. Exactly one
    /// of the two halves is populated per layer; a layer that is neither
    /// is a construction error, not a runtime fallback.
    pub fn new(
        device: Vec<Option<DeviceLayer>>,
        host: Vec<Option<(LayerSpec<'a>, LayerState)>>,
    ) -> Self {
        assert_eq!(device.len(), host.len(), "one slot per layer, in order");
        for (i, (d, h)) in device.iter().zip(&host).enumerate() {
            assert!(
                d.is_some() ^ h.is_some(),
                "layer {i} must run in exactly one place"
            );
        }
        Self {
            device,
            host,
            head: None,
        }
    }

    /// Attach the head so it runs INSIDE the last device epoch.
    ///
    /// Returns `false` — and attaches nothing — unless the stack ends
    /// on a device layer, because a head appended to a command buffer
    /// that is not last would read a hidden state the host still has to
    /// finish. Refusing is the point: silently running the head on the
    /// host instead would leave the caller measuring the CPU path while
    /// believing it had moved.
    pub fn attach_head(&mut self, head: HybridHead) -> bool {
        if !matches!(self.device.last(), Some(Some(_))) {
            return false;
        }
        self.head = Some(head);
        true
    }

    /// Whether [`Self::forward`] returns logits rather than a hidden
    /// state.
    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Maximal runs of consecutive device layers — one command buffer
    /// each.
    pub fn epochs(&self) -> Vec<std::ops::Range<usize>> {
        let mut runs = Vec::new();
        let mut start: Option<usize> = None;
        for i in 0..self.device.len() {
            match (self.device[i].is_some(), start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    runs.push(s..i);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            runs.push(s..self.device.len());
        }
        runs
    }

    /// One token through the whole mixed stack.
    ///
    /// Device runs go out as one command buffer each; host layers run
    /// through the proven CPU path. The hidden state crosses only at the
    /// boundaries between them.
    pub fn forward(
        &mut self,
        metal: &MetalBackend,
        x: &[f32],
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<StackLayerTrace>, HybridTiming), GroupedError> {
        self.forward_traced(metal, x, hidden, None)
    }

    /// The same forward, optionally collecting each device layer's
    /// routing decision.
    ///
    /// `None` is serving: the instrumentation must not leak into the hot
    /// path, so it is a parameter rather than a field that is always
    /// there and merely unused.
    pub fn forward_traced(
        &mut self,
        metal: &MetalBackend,
        x: &[f32],
        hidden: usize,
        mut trace: Option<&mut ExecutionTrace>,
    ) -> Result<(Vec<f32>, Vec<StackLayerTrace>, HybridTiming), GroupedError> {
        let mut h = x.to_vec();
        let mut traces = Vec::with_capacity(self.device.len());
        let mut t = HybridTiming::default();
        let mut i = 0usize;
        while i < self.device.len() {
            if self.device[i].is_some() {
                let end = (i..self.device.len())
                    .take_while(|&j| self.device[j].is_some())
                    .last()
                    .expect("at least i")
                    + 1;
                let calls: Vec<KimiLayerCall<'_>> = self.device[i..end]
                    .iter()
                    .map(|d| KimiLayerCall {
                        weights: d.as_ref().expect("device run").weights(),
                    })
                    .collect();
                let clock = Instant::now();
                // The head rides in the LAST epoch's command buffer, so
                // the hidden state it consumes never crosses to the host
                // and the token costs no extra submission.
                let last_run = end == self.device.len();
                let (out, gpu) = match (&self.head, last_run) {
                    (Some(hd), true) => metal.kimi_decoder_layers_with_head(
                        &calls,
                        &KimiHead {
                            norm_weight: &hd.norm_weight,
                            norm_eps: hd.norm_eps,
                            weight: &hd.weight,
                            vocab: hd.vocab,
                        },
                        &h,
                        trace.as_deref_mut(),
                    )?,
                    _ => metal.kimi_decoder_layers(&calls, &h, trace.as_deref_mut())?,
                };
                t.device_wall_ms += clock.elapsed().as_secs_f64() * 1000.0;
                t.device_gpu_ms += gpu;
                t.epochs += 1;
                h = out;
                i = end;
            } else {
                let (spec, state) = self.host[i].as_mut().expect("host layer");
                let clock = Instant::now();
                let trace = layer_forward_public(i, &h, hidden, spec, state);
                t.host_wall_ms += clock.elapsed().as_secs_f64() * 1000.0;
                h = trace.layer_output.clone();
                traces.push(trace);
                i += 1;
            }
        }
        Ok((h, traces, t))
    }
}
