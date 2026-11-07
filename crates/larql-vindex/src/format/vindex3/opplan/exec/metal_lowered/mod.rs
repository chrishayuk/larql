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
//! Lives beside the other Metal executors (`stack_metal`, `kda_metal`),
//! behind the same `gpu` + macOS gate: this is the one place a plan and a
//! device are both in scope, which keeps the lowering primitives
//! (`larql-compute-metal`) free of plan types and the plan free of device
//! types. It used to live in the CLI, on the reasoning that `larql-vindex`
//! never linked Metal; it has since `gpu` gained `larql-compute-metal`, and
//! the session is library code every front-end can drive. The drivers —
//! argument handling, dumps, speculative decode — stay in the CLI.

use std::collections::HashMap;

use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::ComponentOpPlan;
use larql_compute_metal::lowering::ffn::FfnActivation;
use larql_compute_metal::lowering::{DeviceBuffer, LoweredMatrix};
use larql_compute_metal::MetalBackend;

/// One matrix operand, resident on the device.
pub mod measure_arm;
mod profile;
mod resident;
mod routed;
mod step;
#[cfg(test)]
mod tests;
mod verify;

pub use step::host_argmax;

use profile::{StageBytes, StageLedger};
use resident::Ablation;
use routed::FfnResident;

mod construct;
mod layer_lowering;

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
    /// `Some` while `--profile` is recording VERIFY-N blocks — a separate
    /// ledger, since a block is not a token.
    verify_ledger: Option<StageLedger>,
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
    /// Command buffers committed, cumulative — a look-ahead that is later
    /// discarded included, since the device ran it. Counted where `commit`
    /// is called, so a submission rate is observed, never inferred from
    /// the one-buffer-per-token design.
    submissions: u64,
    /// The 18 stack scratch slot widths (floats per position), kept so a
    /// verify block can allocate the same slots `rows` positions deep.
    scratch_widths: Vec<usize>,
    /// VERIFY-N scratch, allocated on first use at the widest block seen
    /// (see `verify.rs`).
    verify: Option<verify::VerifyScratch>,
    /// SPLITK-1 attention partials `(o, m/l)` for one position, and the
    /// widest op's `(num_q_heads, num_q_heads * head_dim)` they are sized
    /// for (a verify block sizes its own from the same pair).
    splitk: [DeviceBuffer; 2],
    splitk_widths: (usize, usize),
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
        self.verify_ledger = Some(StageLedger::default());
    }

    /// The VERIFY-N block ledger, rendered per block; `None` if profiling
    /// never started or no block was verified.
    pub fn verify_profile_report(&self) -> Option<Vec<String>> {
        let ledger = self.verify_ledger.as_ref().filter(|l| l.tokens > 0)?;
        let mut lines = ledger.render();
        if let Some(first) = lines.first_mut() {
            *first = first
                .replace("token(s)", "verify block(s)")
                .replace("per token", "per verify block");
        }
        Some(lines)
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
            if let FfnResident::Dense { gate, up, down } = &l.ffn {
                b.dense_gate_up += gate.bytes() + up.bytes();
                b.dense_down += down.bytes();
            }
        }
        if let Some(h) = &self.head {
            b.head = h.bytes();
        }
        b
    }

    /// The residual width every position carries.
    pub fn hidden(&self) -> usize {
        self.hidden
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

/// The lowering's gate/up combine for the plan's, or why there is none.
///
/// Reads the POLICY first: a policy that is not plain gating owns the
/// whole combine and the nonlinearity beside it is inert, so asking the
/// activation first would answer for a field the layer never reads.
fn ffn_activation(
    activation: larql_models::config::Activation,
    gate_policy: larql_models::ExpertGatePolicy,
) -> Result<FfnActivation, VindexError> {
    use larql_models::config::Activation;
    match gate_policy {
        larql_models::ExpertGatePolicy::SituGlu { beta, linear_beta } => {
            return Ok(FfnActivation::SituGlu { beta, linear_beta })
        }
        larql_models::ExpertGatePolicy::ClampedGlu { limit, alpha } => {
            return Err(VindexError::Parse(format!(
                "the lowering has no gate/up kernel for ExpertGatePolicy::ClampedGlu \
                 {{ limit: {limit}, alpha: {alpha} }} (A-9.4); refusing rather than lowering \
                 it as plain gating"
            )))
        }
        larql_models::ExpertGatePolicy::ClampedGated { limit } => {
            return Err(VindexError::Parse(format!(
                "the lowering has no gate/up kernel for ExpertGatePolicy::ClampedGated \
                 {{ limit: {limit} }}; refusing rather than lowering it as plain gating, \
                 whose clamp is one-sided on the gate and symmetric on the up branch"
            )))
        }
        larql_models::ExpertGatePolicy::Gated => {}
    }
    match activation {
        Activation::Silu => Ok(FfnActivation::Silu),
        Activation::GeluTanh => Ok(FfnActivation::GeluTanh),
        other => Err(VindexError::Parse(format!(
            "the lowering has no gate/up kernel for activation {other:?}; refusing"
        ))),
    }
}
