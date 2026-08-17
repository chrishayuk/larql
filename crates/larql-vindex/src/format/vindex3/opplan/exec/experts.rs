//! Loading a layer's FFN operands — dense or routed — in the backend's
//! declared format, and building the resolved call.
//!
//! The routed case binds a **packed expert bank**: every expert's
//! projections live in one operand (`[experts, rows, k]`, MXFP4 blocks
//! plus a scales stream, or unquantised BF16). Per expert, the loader
//! either binds the stored bytes as they are — an MXFP4 expert for a
//! backend that declared MXFP4 is a copy into aligned memory and nothing
//! else — or converts through f32 to the format the backend asked for,
//! exactly as `load_weight` does for a dense matrix. One resolution path,
//! so the batch executor and the decode session cannot drift.

use larql_models::config::ExpertFormat;
use larql_models::quant::mxfp4::{dequantize_expert, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::backend::{FfnCall, RoutedFfnCall, WeightFormat, WeightSlice};
use super::operands::{widen, OperandStore};
use super::weights::{
    bf16_bytes_to_f16, f32_bytes_to_f16, load_weight, quantize_mxfp4, quantize_nvfp4, AlignedBytes,
    LoadedWeight,
};
use crate::error::VindexError;
use crate::format::vindex3::opplan::{LayerFfn, PackedProjection, RoutedFfnOp};

/// Stored dtype of MXFP4 block and scale streams.
const DTYPE_U8: &str = "U8";
/// Stored dtype of an unquantised packed bank.
const DTYPE_BF16: &str = "BF16";
/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// A layer's FFN operands, loaded once in the backend's declared format.
pub(super) enum FfnOperands {
    Dense {
        gate: Option<LoadedWeight>,
        up: LoadedWeight,
        down: LoadedWeight,
    },
    Routed(RoutedOperands),
}

/// A routed layer's operands: router (f32 glue) and per-expert matrices.
pub(super) struct RoutedOperands {
    router: Vec<f32>,
    router_bias: Option<Vec<f32>>,
    gate_up: Vec<LoadedWeight>,
    gate_up_bias: Option<Vec<f32>>,
    down: Vec<LoadedWeight>,
    down_bias: Option<Vec<f32>>,
}

impl FfnOperands {
    pub(super) fn load(
        ffn: &LayerFfn,
        store: &OperandStore,
        format: WeightFormat,
    ) -> Result<Self, VindexError> {
        match ffn {
            LayerFfn::Dense(op) => Ok(Self::Dense {
                gate: match &op.gate {
                    Some(gate) => Some(load_weight(store, gate, format)?),
                    None => None,
                },
                up: load_weight(store, &op.up, format)?,
                down: load_weight(store, &op.down, format)?,
            }),
            LayerFfn::Routed(op) => Ok(Self::Routed(RoutedOperands::load(op, store, format)?)),
        }
    }

    /// Every matrix operand, for residency preparation.
    pub(super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Dense { gate, up, down } => {
                let mut slices = vec![up.slice(), down.slice()];
                if let Some(gate) = gate {
                    slices.push(gate.slice());
                }
                slices
            }
            Self::Routed(routed) => routed
                .gate_up
                .iter()
                .chain(&routed.down)
                .map(LoadedWeight::slice)
                .collect(),
        }
    }

    /// Run this layer's FFN over one normalised vector on `backend`.
    pub(super) fn apply<B: super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        x: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        match (self, ffn) {
            (Self::Dense { gate, up, down }, LayerFfn::Dense(op)) => backend.ffn(FfnCall {
                x,
                hidden,
                intermediate: op.intermediate_size,
                gate: gate.as_ref().map(LoadedWeight::slice),
                up: up.slice(),
                down: down.slice(),
                activation: op.activation,
                gate_policy: op.gate_policy,
            }),
            (Self::Routed(routed), LayerFfn::Routed(op)) => {
                let gate_up: Vec<WeightSlice<'_>> =
                    routed.gate_up.iter().map(LoadedWeight::slice).collect();
                let down: Vec<WeightSlice<'_>> =
                    routed.down.iter().map(LoadedWeight::slice).collect();
                backend.routed_ffn(RoutedFfnCall {
                    x,
                    hidden,
                    intermediate: op.expert_intermediate_size,
                    experts: op.experts,
                    top_k: op.top_k,
                    router_kind: op.router_kind,
                    routing_policy: op.routing_policy,
                    activation: op.activation,
                    gate_policy: op.gate_policy,
                    gate_up_layout: op.gate_up_layout.ok_or_else(|| {
                        VindexError::Parse(
                            "routed FFN op carries no gate_up layout; closure requires one"
                                .to_string(),
                        )
                    })?,
                    router: &routed.router,
                    router_bias: routed.router_bias.as_deref(),
                    gate_up: &gate_up,
                    gate_up_bias: routed.gate_up_bias.as_deref(),
                    down: &down,
                    down_bias: routed.down_bias.as_deref(),
                })
            }
            _ => Err(VindexError::Parse(
                "FFN operands were loaded for a different op kind than the plan carries"
                    .to_string(),
            )),
        }
    }
}

impl RoutedOperands {
    fn load(
        op: &RoutedFfnOp,
        store: &OperandStore,
        format: WeightFormat,
    ) -> Result<Self, VindexError> {
        let hidden = op.router.shape.get(1).copied().unwrap_or(0);
        let inter = op.expert_intermediate_size;
        Ok(Self {
            router: store.load(&op.router)?,
            router_bias: op.router_bias.as_ref().map(|b| store.load(b)).transpose()?,
            gate_up: load_packed(
                store,
                &op.gate_up,
                op,
                FUSED_BRANCHES * inter,
                hidden,
                format,
            )?,
            gate_up_bias: op
                .gate_up
                .bias
                .as_ref()
                .map(|b| store.load(b))
                .transpose()?,
            down: load_packed(store, &op.down, op, hidden, inter, format)?,
            down_bias: op.down.bias.as_ref().map(|b| store.load(b)).transpose()?,
        })
    }
}

/// Load one packed projection as `experts` matrices of `[rows, k]` in
/// `format`.
fn load_packed(
    store: &OperandStore,
    projection: &PackedProjection,
    op: &RoutedFfnOp,
    rows: usize,
    k: usize,
    format: WeightFormat,
) -> Result<Vec<LoadedWeight>, VindexError> {
    let name = projection.weights.tensor.as_str();
    let raw = store.load_raw(&projection.weights)?;
    match op.expert_format {
        ExpertFormat::PackedMxfp4 => {
            let scales_ref = projection.scales.as_ref().ok_or_else(|| {
                VindexError::Parse(format!(
                    "`{name}`: MXFP4 expert projection carries no scales operand"
                ))
            })?;
            let scales = store.load_raw(scales_ref)?;
            expect_dtype(&raw.dtype, DTYPE_U8, name)?;
            expect_dtype(&scales.dtype, DTYPE_U8, &scales_ref.tensor)?;
            if !k.is_multiple_of(MXFP4_GROUP_ELEMS) {
                return Err(VindexError::Parse(format!(
                    "`{name}`: k={k} is not a multiple of the MXFP4 group"
                )));
            }
            let groups = k / MXFP4_GROUP_ELEMS;
            let block_stride = rows * groups * MXFP4_GROUP_BYTES;
            let scale_stride = rows * groups;
            expect_len(raw.bytes.len(), op.experts * block_stride, name)?;
            expect_len(
                scales.bytes.len(),
                op.experts * scale_stride,
                &scales_ref.tensor,
            )?;
            (0..op.experts)
                .map(|e| {
                    let packed = &raw.bytes[e * block_stride..(e + 1) * block_stride];
                    let scale = &scales.bytes[e * scale_stride..(e + 1) * scale_stride];
                    match format {
                        // Native: the stored bytes are the operand.
                        WeightFormat::Mxfp4 => Ok(LoadedWeight::Mxfp4 {
                            packed: AlignedBytes::from_bytes(packed),
                            scales: AlignedBytes::from_bytes(scale),
                        }),
                        // Everything else converts through f32, exactly as
                        // a dense matrix would.
                        other => {
                            let values = dequantize_expert(packed, scale, rows, groups)
                                .map_err(|e| VindexError::Parse(format!("`{name}`: {e}")))?;
                            from_f32(values, rows, k, other, name)
                        }
                    }
                })
                .collect()
        }
        ExpertFormat::PackedBF16 => {
            expect_dtype(&raw.dtype, DTYPE_BF16, name)?;
            let stride = rows * k * 2;
            expect_len(raw.bytes.len(), op.experts * stride, name)?;
            (0..op.experts)
                .map(|e| {
                    let bytes = &raw.bytes[e * stride..(e + 1) * stride];
                    match format {
                        WeightFormat::F16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(bytes, name)?)),
                        other => from_f32(widen(DTYPE_BF16, bytes, name)?, rows, k, other, name),
                    }
                })
                .collect()
        }
        ExpertFormat::PerExpert => Err(VindexError::Parse(format!(
            "`{name}`: per-expert tensors are not a packed projection; closure never plans one"
        ))),
    }
}

/// One expert's `[rows, k]` f32 matrix, converted to `format`.
fn from_f32(
    values: Vec<f32>,
    rows: usize,
    k: usize,
    format: WeightFormat,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(values)),
        WeightFormat::F16 => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(LoadedWeight::F16(f32_bytes_to_f16(&bytes, name)?))
        }
        WeightFormat::Mxfp4 => quantize_mxfp4(&values, rows, k, name),
        WeightFormat::Nvfp4 => quantize_nvfp4(&values, rows, k, name),
    }
}

fn expect_dtype(found: &str, expected: &str, name: &str) -> Result<(), VindexError> {
    if found == expected {
        Ok(())
    } else {
        Err(VindexError::Parse(format!(
            "`{name}`: expected stored dtype {expected}, found {found}"
        )))
    }
}

fn expect_len(found: usize, expected: usize, name: &str) -> Result<(), VindexError> {
    if found == expected {
        Ok(())
    } else {
        Err(VindexError::Parse(format!(
            "`{name}`: {found} stored bytes, expected {expected} for the declared expert geometry"
        )))
    }
}
