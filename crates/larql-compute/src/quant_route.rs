//! `FormatRoute` — the per-format compute dispatch record (Roadmap #7).
//!
//! One place that says what this crate can do with a [`QuantFormat`]:
//! which dequant decodes it, which Q8K-activation matvec kernel serves
//! it on the decode path, which amortised matmul serves it at seq>1.
//! The string-keyed dispatch sites (`kquant_forward::dequant`,
//! `q4k_q8k_matvec_parallel`, `ffn::weight::quant_matmul`,
//! `kquant_forward::cached`) resolve `tag → QuantFormat → FormatRoute`
//! here instead of each maintaining its own `"Q4_K" | "Q6_K"` match.
//!
//! **Adding a format** (MXFP4 experts at DEC-6a, BF16 slabs) = one
//! [`QuantFormat`] variant + one arm in [`QuantFormat::route`] + the
//! kernels it points at. Formats whose weights carry side metadata
//! (ternary I2_S's per-channel scale array; MXFP4's E8M0 scales) do not
//! fit the flat `&[u8]` block-stream signatures — they follow the
//! `ternary_matvec` parallel-path template and leave these fields
//! `None` (see `cpu/ops/ternary_matvec.rs`).
//!
//! ## Unknown-format contract
//!
//! - **Seam parsers** (`pipeline_layer::{attn,ffn}_str_to_format`,
//!   `QuantFormat::from_registry_tag`) reject unknown tags loudly at the
//!   boundary.
//! - **Capability probes** (`FormatRoute` field `is_some()`,
//!   `layer_supports_direct_matvec`, `supports_quant`) return
//!   `false`/`None` so the caller takes its dequant fallback.
//! - **Kernels never silently zero.** A kernel entry point handed a
//!   format it has no route for panics with the offending tag — a wrong
//!   number with no error is the failure mode the DEC programme can
//!   least afford (dec-readiness review, §1).

use crate::cpu::ops::q4_common::{q4k_matmul_into, q6k_matmul_into};
use crate::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_into, q6k_q8k_matvec_into, Q8KActivation};
use crate::cpu::ops::KernelShapeError;
use crate::QuantFormat;

/// Dequantise exactly `padded_elems` (a block-multiple) values from a
/// packed row-major byte stream.
pub type DequantPaddedFn = fn(&[u8], usize) -> Result<Vec<f32>, String>;

/// Multi-row matvec `(out, q8k_x, weight_bytes, rows, cols)` against a
/// pre-quantised Q8K activation. Single-threaded per call; row-chunk
/// parallelism lives in `q4k_q8k_matvec_parallel`.
pub type Q8kMatvecFn =
    fn(&mut [f32], &Q8KActivation, &[u8], usize, usize) -> Result<(), KernelShapeError>;

/// Amortised matmul `(out, x, weight_bytes, rows, cols, seq)` reading
/// the packed bytes once for all `seq` positions.
pub type QuantMatmulFn = fn(&mut [f32], &[f32], &[u8], usize, usize, usize);

/// What the CPU compute path can do with one [`QuantFormat`].
///
/// Every field is an optional capability: `None` means "no direct
/// kernel — dequantise and take the f32 path" (a probe result, never an
/// error). Block geometry stays on [`QuantFormat::packed_block_layout`].
#[derive(Clone, Copy)]
pub struct FormatRoute {
    /// The format this route serves.
    pub format: QuantFormat,
    /// Padded row-major dequant (the `dequantize_matrix` backend).
    pub dequant_padded: Option<DequantPaddedFn>,
    /// Decode-path matvec against a Q8K-quantised activation.
    pub q8k_matvec: Option<Q8kMatvecFn>,
    /// Prefill/seq>1 amortised matmul over f32 activations.
    pub quant_matmul: Option<QuantMatmulFn>,
}

impl FormatRoute {
    const fn empty(format: QuantFormat) -> Self {
        Self {
            format,
            dequant_padded: None,
            q8k_matvec: None,
            quant_matmul: None,
        }
    }
}

fn dequant_q4k_padded(bytes: &[u8], padded: usize) -> Result<Vec<f32>, String> {
    larql_models::quant::ggml::q4_k::dequantize_q4_k(bytes, padded).map_err(|e| e.to_string())
}

fn dequant_q6k_padded(bytes: &[u8], padded: usize) -> Result<Vec<f32>, String> {
    larql_models::quant::ggml::q6_k::dequantize_q6_k(bytes, padded).map_err(|e| e.to_string())
}

impl QuantFormat {
    /// The per-format dispatch record. One match arm per format — the
    /// single place a new format registers its CPU kernels.
    pub fn route(self) -> FormatRoute {
        match self {
            Self::Q4_K => FormatRoute {
                format: self,
                dequant_padded: Some(dequant_q4k_padded),
                q8k_matvec: Some(q4k_q8k_matvec_into),
                quant_matmul: Some(q4k_matmul_into),
            },
            Self::Q6_K => FormatRoute {
                format: self,
                dequant_padded: Some(dequant_q6k_padded),
                q8k_matvec: Some(q6k_q8k_matvec_into),
                quant_matmul: Some(q6k_matmul_into),
            },
            // Q4_KF/Q4_0/Q8_0 flow through the `QuantMatVec` trait's
            // legacy dispatch; floats and side-metadata formats (I2S)
            // have no block-stream kernels here by design.
            other => FormatRoute::empty(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    #[test]
    fn kquant_routes_have_all_capabilities() {
        for fmt in [QuantFormat::Q4_K, QuantFormat::Q6_K] {
            let route = fmt.route();
            assert_eq!(route.format, fmt);
            assert!(route.dequant_padded.is_some(), "{fmt:?} dequant");
            assert!(route.q8k_matvec.is_some(), "{fmt:?} q8k matvec");
            assert!(route.quant_matmul.is_some(), "{fmt:?} matmul");
        }
    }

    #[test]
    fn non_kquant_routes_are_empty_probes_not_errors() {
        for fmt in [
            QuantFormat::Q4_0,
            QuantFormat::Q4_KF,
            QuantFormat::Q8_0,
            QuantFormat::BF16,
            QuantFormat::F16,
            QuantFormat::F32,
            QuantFormat::I2S,
        ] {
            let route = fmt.route();
            assert_eq!(route.format, fmt);
            assert!(route.dequant_padded.is_none(), "{fmt:?}");
            assert!(route.q8k_matvec.is_none(), "{fmt:?}");
            assert!(route.quant_matmul.is_none(), "{fmt:?}");
        }
    }

    /// The route's dequant fns are the same implementations
    /// `dequantize_matrix` always called — pin that equivalence.
    #[test]
    fn route_dequant_matches_direct_kernel_calls() {
        let n = 256;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.01).collect();

        let q4k = quantize_q4_k(&values);
        let via_route = (QuantFormat::Q4_K.route().dequant_padded.unwrap())(&q4k, n).unwrap();
        let direct = larql_models::quant::ggml::q4_k::dequantize_q4_k(&q4k, n).unwrap();
        assert_eq!(via_route, direct);

        let q6k = quantize_q6_k(&values);
        let via_route = (QuantFormat::Q6_K.route().dequant_padded.unwrap())(&q6k, n).unwrap();
        let direct = larql_models::quant::ggml::q6_k::dequantize_q6_k(&q6k, n).unwrap();
        assert_eq!(via_route, direct);
    }
}
