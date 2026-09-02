//! Resident device operands (matrices, norms, rope tables) and the
//! ablation flags the lowered session binds — loaded once, held for the
//! session's lifetime.

use larql_compute_metal::lowering::DeviceBuffer;
use larql_compute_metal::MetalBackend;
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::{
    load_weight, AlignedBytes, LoadedWeight,
};
use larql_vindex::format::vindex3::opplan::{NormOp, OperandRef};

use super::DeviceMatrix;

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
pub(super) struct Ablation {
    pub no_query_scale: bool,
    pub no_rope: bool,
    pub no_qk_norm: bool,
    pub no_gate: bool,
    pub no_post_norms: bool,
}

impl Ablation {
    pub(super) fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok();
        Self {
            no_query_scale: on("LARQL_ABLATE_QUERY_SCALE"),
            no_rope: on("LARQL_ABLATE_ROPE"),
            no_qk_norm: on("LARQL_ABLATE_QK_NORM"),
            no_gate: on("LARQL_ABLATE_GATE"),
            no_post_norms: on("LARQL_ABLATE_POST_NORMS"),
        }
    }

    pub(super) fn any(&self) -> bool {
        self.no_query_scale || self.no_rope || self.no_qk_norm || self.no_gate || self.no_post_norms
    }
}

/// Load one matrix operand as NVFP4 and hand it to the device.
///
/// The buffers are keyed on the `AlignedBytes` address, which lives for
/// the session, so `lowering_weight` caches them and the weight is
/// uploaded once rather than per position.
///
/// `read_bytes` counts `logical_len`, not `as_slice().len()`. The
/// allocation is padded to `DEVICE_PAGE_ALIGN`; the *tensor* is the prefix,
/// and the byte-floor ledger divides these counts by measured bandwidth to
/// price a token. Counting padding would inflate both the bytes and the
/// achieved GB/s that follow from them.
///
/// The error is shape-dependent and therefore treacherous: it is exactly
/// zero on Granite (every allocation lands on a page boundary) and 0.43% on
/// gpt-oss's 2880-wide stack, so a ledger validated on one model can drift
/// silently on the next. `packed_attention` below already counted logical
/// bytes; these three arms did not.
pub(super) fn resident_matrix(
    gpu: &MetalBackend,
    store: &OperandStore,
    operand: &OperandRef,
    format: WeightFormat,
    keep: &mut Vec<LoadedWeight>,
) -> Result<DeviceMatrix, VindexError> {
    let rows = operand.shape.first().copied().unwrap_or(0);
    let cols = operand.shape.get(1).copied().unwrap_or(0);
    let loaded = load_weight(store.into(), operand, format)?;
    let m = match &loaded {
        LoadedWeight::Nvfp4 {
            packed,
            scales,
            tensor_scale,
        } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: packed.logical_len() + scales.logical_len(),
            tensor_scale: *tensor_scale,
            format: WeightFormat::Nvfp4,
            rows,
            cols,
        },
        LoadedWeight::Mxfp4 { packed, scales } => DeviceMatrix {
            packed: gpu.lowering_weight(packed.as_slice()),
            scales: gpu.lowering_weight(scales.as_slice()),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: packed.logical_len() + scales.logical_len(),
            tensor_scale: 1.0,
            format: WeightFormat::Mxfp4,
            rows,
            cols,
        },
        LoadedWeight::F16(bytes) => DeviceMatrix {
            packed: gpu.lowering_weight(bytes.as_slice()),
            scales: gpu.lowering_weight(&[]),
            packed_offset: 0,
            scales_offset: 0,
            read_bytes: bytes.logical_len(),
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

/// Operator control for the attention single-allocation rung. The name
/// predates O joining the pack.
///
/// * `0` — one allocation per matrix (the baseline arm)
/// * `qkv` — Q, K and V share an allocation; O keeps its own
/// * unset / anything else — all four share, in touch order
///
/// The three-way exists so O's contribution is a DIRECT measurement.
/// O cannot join the fused Q/K/V dispatch (its `k` differs), so the
/// `qkv` → all step changes co-location and nothing else; reading it as
/// the difference of two separately-run two-way A/Bs would subtract
/// across machine states instead. Read once.
pub(super) const QKV_PACK_ENV: &str = "LARQL_QKV_PACK";

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackArity {
    None,
    Qkv,
    All,
}

impl PackArity {
    /// Operands this arity co-locates, out of `ATTN_PACK_OPERANDS`.
    fn packed(self) -> usize {
        match self {
            PackArity::None => 0,
            PackArity::Qkv => 3,
            PackArity::All => ATTN_PACK_OPERANDS,
        }
    }
}

fn pack_arity() -> PackArity {
    static ARITY: std::sync::OnceLock<PackArity> = std::sync::OnceLock::new();
    *ARITY.get_or_init(|| match std::env::var(QKV_PACK_ENV).as_deref() {
        Ok("0") => PackArity::None,
        Ok("qkv") => PackArity::Qkv,
        _ => PackArity::All,
    })
}

/// Metal bind alignment the packed slices must land on: the x2 body
/// loads codes as `uint2`, so a slice's base (buffer offset included)
/// must be 16-byte aligned; scales are held to the same bound.
const PACK_OFFSET_ALIGN: usize = 16;

/// Attention operands the pack co-locates, in the order a token touches
/// them: Q, K and V (one fused dispatch), then O.
const ATTN_PACK_OPERANDS: usize = 4;

/// Load Q, K, V and O resident as slices of ONE codes allocation and ONE
/// scales allocation, laid out in the order the token consumes them.
///
/// Two distinct effects live here and they are worth keeping apart. For
/// Q/K/V the pack also makes the FUSED projection dispatch stream a
/// single contiguous range (`examples/qkv_seg3_probe.rs` arm D measured
/// that at 1.6 us/layer). O cannot join that dispatch at all — its `k`
/// is `num_q_heads * head_dim`, not `hidden`, and a fused segment must
/// share `k` — so O gains NOTHING at the kernel level: same kernel, same
/// bytes, same grid, only a different base address. Any end-to-end
/// movement attributable to O is therefore pure in-situ locality, which
/// is exactly the hypothesis the A-12 in-situ deficit poses (stages
/// repeated 24x read 12-19% under their isolated bandwidth while the
/// once-per-token head reads 96% of it).
///
/// Packing applies only when the attention class is NVFP4 and every
/// slice boundary meets the bind alignment; any other case loads the
/// matrices separately, unchanged.
pub(super) fn resident_attn(
    gpu: &MetalBackend,
    store: &OperandStore,
    ops: [&OperandRef; ATTN_PACK_OPERANDS],
    format: WeightFormat,
    keep: &mut Vec<LoadedWeight>,
) -> Result<[DeviceMatrix; ATTN_PACK_OPERANDS], VindexError> {
    let unpacked = |gpu, store, keep: &mut Vec<LoadedWeight>| {
        let mut out = Vec::with_capacity(ATTN_PACK_OPERANDS);
        for op in ops {
            out.push(resident_matrix(gpu, store, op, format, keep)?);
        }
        <[DeviceMatrix; ATTN_PACK_OPERANDS]>::try_from(out)
            .map_err(|_| VindexError::Parse("attention operand set has the wrong arity".into()))
    };
    // The fusion flag gates only the Q/K/V dispatch, not the layout; a
    // run with fusion off still benefits from co-location, and keeping
    // the two independent is what lets the A/B separate them.
    let arity = pack_arity();
    if format != WeightFormat::Nvfp4 || arity == PackArity::None {
        return unpacked(gpu, store, keep);
    }
    // Operands beyond the arity keep their own allocations, so the
    // `qkv` arm differs from the `all` arm in co-location ONLY.
    let n_packed = arity.packed();
    let mut loaded = Vec::with_capacity(n_packed);
    for op in ops.iter().take(n_packed) {
        loaded.push(load_weight(store.into(), op, format)?);
    }
    let parts: Vec<(&AlignedBytes, &AlignedBytes, f32)> = loaded
        .iter()
        .filter_map(|w| match w {
            LoadedWeight::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => Some((packed, scales, *tensor_scale)),
            _ => None,
        })
        .collect();
    // NVFP4 was requested, so each loads as NVFP4; anything else is a
    // load_weight contract change — fall back rather than misbind.
    if parts.len() != n_packed {
        return unpacked(gpu, store, keep);
    }
    // Slice boundaries are cumulative LOGICAL lengths; every boundary
    // must meet the bind alignment or a dispatch would misread rows.
    let aligned = parts.iter().all(|(p, s, _)| {
        p.logical_len() % PACK_OFFSET_ALIGN == 0 && s.logical_len() % PACK_OFFSET_ALIGN == 0
    });
    if !aligned {
        return unpacked(gpu, store, keep);
    }
    let mut packed_all = Vec::with_capacity(parts.iter().map(|(p, ..)| p.logical_len()).sum());
    let mut scales_all = Vec::with_capacity(parts.iter().map(|(_, s, _)| s.logical_len()).sum());
    let mut offsets = vec![(0u64, 0u64); n_packed];
    for (i, (p, s, _)) in parts.iter().enumerate() {
        offsets[i] = (packed_all.len() as u64, scales_all.len() as u64);
        packed_all.extend_from_slice(&p.as_slice()[..p.logical_len()]);
        scales_all.extend_from_slice(&s.as_slice()[..s.logical_len()]);
    }
    let packed_all = AlignedBytes::from_bytes(&packed_all);
    let scales_all = AlignedBytes::from_bytes(&scales_all);
    let packed_buf = gpu.lowering_weight(packed_all.as_slice());
    let scales_buf = gpu.lowering_weight(scales_all.as_slice());
    let mut out = Vec::with_capacity(ATTN_PACK_OPERANDS);
    for (i, (p, s, tensor_scale)) in parts.iter().enumerate() {
        out.push(DeviceMatrix {
            packed: packed_buf.clone(),
            scales: scales_buf.clone(),
            packed_offset: offsets[i].0,
            scales_offset: offsets[i].1,
            read_bytes: p.logical_len() + s.logical_len(),
            tensor_scale: *tensor_scale,
            format: WeightFormat::Nvfp4,
            rows: ops[i].shape.first().copied().unwrap_or(0),
            cols: ops[i].shape.get(1).copied().unwrap_or(0),
        });
    }
    // Operands outside the pack load as ordinary whole-buffer residents.
    for op in ops.iter().skip(n_packed) {
        out.push(resident_matrix(gpu, store, op, format, keep)?);
    }
    // The device buffers alias the COMBINED allocations; the per-matrix
    // loads can drop, the pack owns the bytes for the session.
    keep.push(LoadedWeight::Nvfp4 {
        packed: packed_all,
        scales: scales_all,
        tensor_scale: 1.0,
    });
    <[DeviceMatrix; ATTN_PACK_OPERANDS]>::try_from(out)
        .map_err(|_| VindexError::Parse("attention pack produced a wrong-arity set".into()))
}

/// Upload an optional f32 vector operand (a bias or the sink logits) to
/// the device, or `None` when the plan carries none.
pub(super) fn resident_vector(
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
pub(super) fn rope_table_key(position: &PositionPolicy, head_dim: usize) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    // The table is `head_dim/2` entries of `theta^(-2i/head_dim)`: two
    // layers at one theta but different head widths (Gemma 4's 256 vs
    // 512) need different tables, so the width is part of every key.
    let with_width = |discriminant: u64| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        discriminant.hash(&mut h);
        head_dim.hash(&mut h);
        h.finish() | 1
    };
    match position {
        PositionPolicy::Rope { theta } => Some(with_width(theta.to_bits())),
        // Llama-3's table is the wavelength-band adjustment of the base
        // frequencies, so every parameter that moves a band is part of
        // the key. Keying on `theta` alone would let two layers scaled
        // for different pre-trained windows share one table.
        PositionPolicy::Llama3 { theta, scaling } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            theta.to_bits().hash(&mut h);
            scaling.factor.to_bits().hash(&mut h);
            scaling.low_freq_factor.to_bits().hash(&mut h);
            scaling.high_freq_factor.to_bits().hash(&mut h);
            scaling
                .original_max_position_embeddings
                .to_bits()
                .hash(&mut h);
            head_dim.hash(&mut h);
            Some(h.finish() | 1)
        }
        // The partial rotary's table is the full-head rotate-half table
        // with the top frequencies zero (head-width basis); fraction and
        // basis join the key.
        PositionPolicy::PartialRope {
            theta,
            rotary_fraction,
            basis,
        } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            theta.to_bits().hash(&mut h);
            rotary_fraction.to_bits().hash(&mut h);
            (*basis == RotaryFrequencyBasis::HeadWidth).hash(&mut h);
            head_dim.hash(&mut h);
            Some(h.finish() | 1)
        }
        // Fold the yarn block into the key so two different blocks (or a
        // block vs plain rope) at one theta get their own tables. The
        // block's f64 fields hash deterministically.
        PositionPolicy::Yarn { theta, scaling } => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            theta.to_bits().hash(&mut h);
            head_dim.hash(&mut h);
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
        // No lowered table: refused in `LoweredSession::new`, so a key
        // is never asked for. `None` rather than a hash keeps a
        // half-built session from sharing a table with plain rope.
        PositionPolicy::MRope { .. } => None,
        // A relative scheme has no rotary table to key, and no lowering
        // consumes it. `None` here means "no table", which is the truthful
        // answer for a policy that does not rotate; execution refuses
        // separately rather than running unpositioned.
        PositionPolicy::Relative { .. } | PositionPolicy::None => None,
    }
}

/// The inverse-frequency table for a rotary policy, matching the
/// interpreter kernel exactly: plain `theta^(-2i/d)` for rope, the YaRN
/// ramp for a scaled layer.
pub(super) fn rope_inv_freq_table(position: &PositionPolicy, head_dim: usize) -> Vec<f32> {
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
        PositionPolicy::Llama3 { theta, scaling } => {
            larql_vindex::format::vindex3::opplan::exec::kernels::llama3_frequencies(
                scaling, head_dim, *theta,
            )
            .iter()
            .map(|f| *f as f32)
            .collect()
        }
        PositionPolicy::Relative { .. } | PositionPolicy::None => Vec::new(),
        // Head-width basis: the interpreter's own table (zeros above the
        // fraction → identity rotation on those pairs). The rotary-width
        // basis rotates a prefix as its own block, which the rope kernel
        // does not express — refused in `LoweredSession::new`.
        PositionPolicy::PartialRope {
            theta,
            rotary_fraction,
            basis: RotaryFrequencyBasis::HeadWidth,
        } => larql_vindex::format::vindex3::opplan::exec::kernels::partial_rotary_frequencies(
            head_dim,
            *rotary_fraction,
            *theta,
        )
        .iter()
        .map(|f| *f as f32)
        .collect(),
        PositionPolicy::PartialRope {
            basis: RotaryFrequencyBasis::RotaryWidth,
            ..
        } => unreachable!("RotaryWidth partial rotary is refused before the session is built"),
        PositionPolicy::MRope { .. } => {
            unreachable!("M-RoPE is refused before the session is built")
        }
    }
}

pub(super) fn resident_norm(
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
