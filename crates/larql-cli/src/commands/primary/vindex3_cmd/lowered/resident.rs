//! Resident device operands (matrices, norms, rope tables) and the
//! ablation flags the lowered session binds — loaded once, held for the
//! session's lifetime.

use larql_compute_metal::lowering::DeviceBuffer;
use larql_compute_metal::MetalBackend;
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};
use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
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
pub(super) fn resident_matrix(
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
        PositionPolicy::None => None,
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
        PositionPolicy::None => Vec::new(),
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
