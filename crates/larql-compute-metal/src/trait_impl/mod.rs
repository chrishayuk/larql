//! `MetalBackend`'s `ComputeBackend`-family trait implementations.
//!
//! One file per sub-trait — mirrors the `backend/` split. The umbrella
//! `ComputeBackend` impl (`name`, `device_info`, `supports`) lives
//! here; sub-trait impls are in their own files.

mod decode;
mod matmul;
mod quant_matvec;

use super::*;
use larql_compute::backend::{Capability, ComputeBackend};

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal (GPU)"
    }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        // Metal accelerates everything in the menu.
        matches!(
            cap,
            Capability::F32Gemv
                | Capability::F16Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::Q4PairBatch
                | Capability::FullPipelineQ4
                | Capability::MultiLayerQ4Ffn
                | Capability::DecodeToken
                | Capability::DecodeMoe
                | Capability::DecodeQ4KMoe
                | Capability::DecodeProfile
                | Capability::PrefillQ4
                | Capability::HeterogeneousAttention
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBackend;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// `name` is the trait-level identifier.  Pin the literal so a
    /// caller switching on `backend.name()` is told when the value
    /// changes (e.g. capitalisation drift).
    #[test]
    fn name_is_stable_identifier() {
        let m = backend();
        assert_eq!(m.name(), "metal (GPU)");
    }

    /// `device_info` includes the FLOP threshold; covering it pins the
    /// fmt string + ensures the accessor stays wired.
    #[test]
    fn device_info_contains_flop_threshold() {
        let m = backend();
        let info = m.device_info();
        assert!(info.starts_with("Metal GPU"));
        assert!(info.contains("FLOP threshold"));
    }

    /// `as_any` returns the same erased reference each call.
    #[test]
    fn as_any_downcasts_back_to_metal_backend() {
        let m = backend();
        let any: &dyn std::any::Any = m.as_any();
        assert!(any.downcast_ref::<MetalBackend>().is_some());
    }

    /// `supports` accepts every capability MetalBackend claims —
    /// exercising every match arm in the `matches!` expression.
    /// Any future variant added to `Capability` will silently default
    /// to `false`; that's the desired conservative behaviour.
    #[test]
    fn supports_every_advertised_capability() {
        let m = backend();
        for cap in [
            Capability::F32Gemv,
            Capability::F16Gemv,
            Capability::QuantMatVec,
            Capability::Q4VecMat,
            Capability::Q4PairBatch,
            Capability::FullPipelineQ4,
            Capability::MultiLayerQ4Ffn,
            Capability::DecodeToken,
            Capability::DecodeMoe,
            Capability::DecodeQ4KMoe,
            Capability::DecodeProfile,
            Capability::PrefillQ4,
            Capability::HeterogeneousAttention,
        ] {
            assert!(m.supports(cap), "{cap:?} should be advertised");
        }
    }
}
