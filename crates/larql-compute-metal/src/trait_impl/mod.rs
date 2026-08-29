//! `MetalBackend`'s `ComputeBackend`-family trait implementations.
//!
//! One file per sub-trait — mirrors the `backend/` split. The umbrella
//! `ComputeBackend` impl (`name`, `device_info`, `supports`) lives
//! here; sub-trait impls are in their own files.

mod bf16_gemv;
pub mod bf16_grouped;
pub mod bf16_moe_block;
mod decode;
pub mod grouped_experts;
pub mod kda;
pub mod kimi_layer;
mod matmul;
pub mod mla;
pub mod mxfp4;
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

    fn register_weight_region(&self, region: &[u8]) {
        // Non-page-aligned bases can't alias zero-copy; resolution simply
        // misses and the MoE dispatch keeps its staged-copy path.
        let _ = self.bufs.register_region(region);
    }

    fn seal_weight_regions(&self) {
        // Declares the registered regions resident once, per
        // `LARQL_RESIDENCY_SET`. Unset => implicit residency, i.e. the
        // behaviour that existed before this hook.
        self.bufs.seal_residency(&self.queue);
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
                | Capability::PerLayerEmbeddings
                | Capability::HybridAttention
        )
    }

    fn prepare_ple_inputs(&self, flat: &[f32], num_layers: usize, ple_dim: usize) {
        MetalBackend::prepare_ple_inputs(self, flat, num_layers, ple_dim);
    }

    fn take_split_timings(&self) -> Option<larql_compute::ProfileTimings> {
        crate::decode::profile::take_last_split_timings()
    }

    fn hybrid_decode_attention_layer(
        &self,
        layer: &larql_compute::FullPipelineLayer<'_>,
        layer_idx: usize,
        x: &[f32],
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        kv_shapes: &[(usize, usize)],
    ) -> Option<Vec<f32>> {
        let mut cache_guard = self.kv_cache_mut_for_shapes(kv_shapes);
        let kv_cache = cache_guard.as_mut()?;
        Some(MetalBackend::decode_attention_layer(
            self, kv_cache, layer, layer_idx, x, hidden, q_dim, kv_dim,
        ))
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

    /// `register_weight_region` is a thin delegator to
    /// `BufferCache::register_region` — pin that the trait method
    /// actually reaches it (a page-aligned anon mmap registers) rather
    /// than silently no-op-ing.
    #[test]
    fn register_weight_region_delegates_to_buffer_cache() {
        let m = backend();
        let mut region = memmap2::MmapMut::map_anon(8192).expect("anon mmap");
        region.fill(0);
        let region = region.make_read_only().expect("read-only mmap");
        let before = m.bufs.region_count();
        m.register_weight_region(&region[..]);
        assert_eq!(m.bufs.region_count(), before + 1);
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
            Capability::PerLayerEmbeddings,
            Capability::HybridAttention,
        ] {
            assert!(m.supports(cap), "{cap:?} should be advertised");
        }
    }

    /// `supports` rejects every capability MetalBackend does NOT
    /// advertise — the `KvDispatch`-family flags (FusedAttentionStep,
    /// WindowedAttentionStep, NativeKvCodec, PipelinedBoundaryUpload,
    /// FusedResidualNorm, KvHandleNative). Those are reserved for the
    /// AsyncComputeBackend Step A6 specialised shaders; Metal returns
    /// `false` for them today so engines route through the CPU-fallback
    /// path.
    #[test]
    fn supports_returns_false_for_unadvertised_kv_dispatch_flags() {
        let m = backend();
        for cap in [
            Capability::FusedAttentionStep,
            Capability::WindowedAttentionStep,
            Capability::NativeKvCodec,
            Capability::PipelinedBoundaryUpload,
            Capability::FusedResidualNorm,
            Capability::KvHandleNative,
        ] {
            assert!(
                !m.supports(cap),
                "{cap:?} must not be advertised until specialised shader lands"
            );
        }
    }

    /// `prepare_ple_inputs` trait override delegates to the inherent
    /// method, which records the buffer + dims on the backend. Reading
    /// back via the internal snapshot confirms the round-trip.
    #[test]
    fn prepare_ple_inputs_trait_dispatch_round_trips_data() {
        let m = backend();
        let num_layers = 4usize;
        let ple_dim = 8usize;
        let data: Vec<f32> = (0..num_layers * ple_dim).map(|i| i as f32).collect();
        ComputeBackend::prepare_ple_inputs(&m, &data, num_layers, ple_dim);
        let snap = m
            .ple_inputs_snapshot()
            .expect("prepare_ple_inputs must populate the cell");
        assert_eq!(snap.num_layers, num_layers);
        assert_eq!(snap.ple_dim, ple_dim);
        assert_eq!(snap.positions, 1);
        m.clear_ple_inputs();
        assert!(m.ple_inputs_snapshot().is_none());
    }

    /// `take_split_timings` trait override reads back the thread-local
    /// the Metal decode path writes to. Direct store + trait take.
    #[test]
    fn take_split_timings_trait_dispatch_returns_stored_value() {
        // Clear any leakage from another test on this thread.
        let _ = crate::decode::profile::take_last_split_timings();

        let m = backend();
        // Without a prior store, the trait method returns None.
        assert!(ComputeBackend::take_split_timings(&m).is_none());

        // After a manual store, the trait method consumes it.
        let written = larql_compute::ProfileTimings {
            attn_ms: 3.0,
            gate_up_ms: 1.5,
            down_ms: 0.5,
            ..Default::default()
        };
        crate::decode::profile::store_last_split_timings(written);
        let read = ComputeBackend::take_split_timings(&m).expect("must surface stored timing");
        assert!((read.attn_ms - 3.0).abs() < 1e-9);
        assert!((read.gate_up_ms - 1.5).abs() < 1e-9);
        assert!((read.down_ms - 0.5).abs() < 1e-9);

        // Consumed — second take is None.
        assert!(ComputeBackend::take_split_timings(&m).is_none());
    }

    /// `hybrid_decode_attention_layer` trait dispatch reaches the
    /// preamble (cache alloc + `as_mut()?`). The full attention
    /// dispatch needs a vindex with `attn_kquant` data already loaded
    /// and a populated KV cache — that end-to-end exercise lives in
    /// `larql-inference`'s `predict_hybrid` integration tests where
    /// the full path is wired against a real `VectorIndex`.
    #[test]
    fn hybrid_decode_attention_layer_trait_dispatch_preamble() {
        let m = backend();
        // Pre-allocate the cache for these shapes; covers `kv_cache_mut_for_shapes`
        // which is the trait method's first line.
        let kv_shapes = [(2usize, 4usize)];
        let guard = m.kv_cache_mut_for_shapes(&kv_shapes);
        assert!(guard.is_some());
        drop(guard);
        // Trait dispatch is reachable on `&dyn ComputeBackend`.
        let any_backend: &dyn ComputeBackend = &m;
        assert!(any_backend.supports(Capability::HybridAttention));
    }

    /// `register_weight_region` + `seal_weight_regions` are the two trait
    /// hooks the serve path calls around model load: register each mmap'd
    /// bank, then declare them resident once. Both are reachable through
    /// `&dyn ComputeBackend`, and sealing must be harmless under the
    /// shipped implicit arm — the composed serve calls it unconditionally.
    #[test]
    fn weight_region_registration_and_sealing_are_reachable_via_the_trait() {
        let m = backend();
        let any_backend: &dyn ComputeBackend = &m;

        // Page-aligned by construction, the production contract's shape.
        let mm = memmap2::MmapMut::map_anon(2 * 16384).expect("anon mmap");
        let region = mm.make_read_only().expect("read-only");
        any_backend.register_weight_region(&region[..]);
        assert_eq!(
            m.bufs.region_count(),
            1,
            "the trait hook must reach BufferCache::register_region"
        );

        // A non-page-aligned base cannot alias zero-copy; the hook
        // swallows the refusal because resolution simply misses later and
        // the MoE dispatch keeps its staged-copy path.
        any_backend.register_weight_region(&region[8..]);
        assert_eq!(m.bufs.region_count(), 1, "a misaligned base adds nothing");

        // Sealing under the shipped arm is a no-op; the queue still works.
        any_backend.seal_weight_regions();
        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/mod.rs:294",
        );
    }
}
