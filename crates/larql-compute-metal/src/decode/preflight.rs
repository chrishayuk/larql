//! Format admission check, run once at decode entry.
//!
//! Every Metal decode path branches on `QuantFormat::is_kquant_family()` to
//! pick between the k-quant route (f32 input, 256-element super-blocks) and
//! the legacy Q8 route. That branch answers "what shape is this weight",
//! not "can this backend serve it" — a format can be a genuine k-quant and
//! still have no shader.
//!
//! Q5_K is the first such format. Its 176-byte super-block would take the
//! k-quant branch and land on the Q4_K kernel's 144-byte stride, decoding
//! into plausible-looking garbage instead of failing. That is the same bug
//! class as routing Q8_0's 34-byte blocks through the Q4_0 kernel
//! (fixed 2026-05-09) — so it gets the same treatment: refuse up front,
//! with a message naming the working path.
//!
//! `stages::quant_matvec::encode` panics on Q5_K too, but that is a
//! backstop several dispatches deep and only on the paths that reach the
//! generic dispatcher. This check runs before any GPU work is encoded.

use larql_compute::{FullPipelineLayer, QuantFormat};

/// Every weight slot a decode layer can carry, with the field name for
/// error reporting. Attention and FFN are listed separately because a
/// mixed vindex can serve Q4_K attention with Q6_K down projections, so
/// the first unservable format is not necessarily in `wq`.
fn layer_formats(layer: &FullPipelineLayer) -> [(&'static str, QuantFormat); 7] {
    [
        ("wq", layer.wq.format),
        ("wk", layer.wk.format),
        ("wv", layer.wv.format),
        ("wo", layer.wo.format),
        ("gate", layer.gate.format),
        ("up", layer.up.format),
        ("down", layer.down.format),
    ]
}

/// Panic if `layer` carries a weight in a format Metal has no kernel for.
///
/// `layer_idx` is included in the message because a mixed-format vindex
/// usually fails on one specific layer, and "which layer" is the first
/// thing you need to rebuild it.
pub fn assert_layer_servable(layer: &FullPipelineLayer, layer_idx: usize) {
    for (slot, format) in layer_formats(layer) {
        assert!(
            format.has_metal_kernel(),
            "metal decode: layer {layer_idx} `{slot}` is {tag}, which has no \
             Metal kernel. {tag} is a 256-element k-quant, so it would \
             otherwise route through the Q4_K/Q6_K path and be read at the \
             wrong super-block stride. Serve this vindex on the CPU backend, \
             or re-extract it as Q4_K/Q6_K.",
            tag = format.registry_tag(),
        );
    }
}

/// Slice form of [`assert_layer_servable`], for the full-pipeline decode
/// entry points. Checks every layer before encoding any GPU work, so a
/// model that is unservable at layer 30 fails immediately rather than
/// after 29 layers of wasted dispatch.
pub fn assert_layers_servable(layers: &[FullPipelineLayer]) {
    for (idx, layer) in layers.iter().enumerate() {
        assert_layer_servable(layer, idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formats Metal actually ships kernels for must all pass, and
    /// Q5_K must not. Pins the admission list itself — if someone adds a
    /// Q5_K shader they should flip `has_metal_kernel` and this test tells
    /// them the guard is now the thing standing in the way.
    #[test]
    fn only_q5k_is_refused() {
        for format in [
            QuantFormat::Q4_0,
            QuantFormat::Q4_K,
            QuantFormat::Q4_KF,
            QuantFormat::Q6_K,
            QuantFormat::Q8_0,
            QuantFormat::BF16,
            QuantFormat::F16,
            QuantFormat::F32,
        ] {
            assert!(
                format.has_metal_kernel(),
                "{format:?} should be admitted by the Metal preflight"
            );
        }
        assert!(
            !QuantFormat::Q5_K.has_metal_kernel(),
            "Q5_K has no Metal shader and must be refused before dispatch"
        );
    }
}
