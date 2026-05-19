//! Native Q4_K/Q6_K FFN walk — fused decode + dot via `kquant_matmul_transb`.
//!
//! Mirrors the inner gated-FFN block of [`crate::vindex::ffn_decode_step_native`]
//! (kquant_forward/cached.rs) but operates as an `FfnBackend`: input is already
//! pre-FFN-normed by the caller (`run_ffn` in `forward/layer.rs`), so this
//! module skips the norm step and just runs the three matmuls + activation.
//!
//! Per-row Q4K decode is fused into the matvec — no whole-layer dequant cache,
//! no f32 staging. On Gemma 3 4B Q4K this is ~5× faster than
//! `walk_ffn_kquant_dequant` because the dequant path materialises three full
//! f32 matrices (gate + up + down, ~120 MB) on every forward call.
//!
//! Position in the routing ladder: priority 4 (after overrides, explicit
//! sparse, and FP4). Inserted before the legacy interleaved Q4_0 / f32 /
//! full_mmap branches because for vindexes that have both Q4K and one of
//! those, native Q4K wins.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    /// Direct Q4_K/Q6_K matvec FFN. Returns `None` when the vindex lacks
    /// Q4K bytes for this layer or the arch isn't a Gated FFN (the only
    /// shape the matvec kernel supports today). Caller falls through to
    /// the next branch.
    pub(super) fn walk_ffn_kquant_native(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        // Gated FFNs only — non-gated archs route through the dequant
        // fallback. Same gate `ffn_decode_step_native` applies.
        let arch = &*self.weights.arch;
        if arch.ffn_type() != larql_models::FfnType::Gated {
            return None;
        }

        // Require Q4K FFN bytes for this layer. `interleaved_kquant_layer_data`
        // returning `None` is the same precondition `kquant_matmul_transb`
        // checks — fail fast rather than letting it report `None` later.
        let _ = self.index.interleaved_kquant_layer_data(layer)?;

        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            // Width must be derivable — see `VectorIndex::num_features`
            // Q4K fallback. A zero here means the manifest entry is
            // unreadable; let the caller fall through.
            return None;
        }

        // Stream next layer's Q4K data while we compute this one — same
        // trick the dequant path uses.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        // Gate (component 0) and up (component 1) — both [intermediate, hidden].
        let x_flat = x.as_slice()?;
        let gate_flat = self
            .index
            .kquant_matmul_transb(layer, 0, x_flat, seq_len, self.backend)?;
        let up_flat = self
            .index
            .kquant_matmul_transb(layer, 1, x_flat, seq_len, self.backend)?;
        let gate = Array2::from_shape_vec((seq_len, intermediate), gate_flat).ok()?;
        let up = Array2::from_shape_vec((seq_len, intermediate), up_flat).ok()?;

        // Element-wise activation. Mirrors `ffn_decode_step_native`'s
        // arch dispatch — GeluTanh for Gemma 3 / Gemma 4, SiLU otherwise.
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate, &up)
        } else {
            crate::ffn::silu_gate_up(&gate, &up)
        };

        // Down (component 2) — [hidden, intermediate], output [seq_len, hidden].
        let act_flat = activation.as_slice()?;
        let down_flat =
            self.index
                .kquant_matmul_transb(layer, 2, act_flat, seq_len, self.backend)?;
        let out = Array2::from_shape_vec((seq_len, hidden), down_flat).ok()?;

        self.trace_path(layer, "interleaved_kquant:native");
        Some((out, activation))
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the Q4_K native walk path. Uses `Q4KTestFixtures`
    //! (hidden=256, intermediate=256, Gemma 3 arch) — the closest in-
    //! process analogue of a real Q4K-only vindex. All assertions
    //! compare against `walk_ffn_kquant_dequant`, which decodes the same
    //! bytes through an independent kernel; matching outputs prove the
    //! native path produces the same FFN output the dequant path does
    //! (both consume the *same* Q4K bytes — equality is the right bar).
    use crate::ffn::FfnBackend;
    use crate::test_utils::Q4KTestFixtures;
    use crate::vindex::WalkFfn;
    use ndarray::Array2;
    use std::sync::OnceLock;

    fn fx() -> &'static Q4KTestFixtures {
        static F: OnceLock<Q4KTestFixtures> = OnceLock::new();
        F.get_or_init(Q4KTestFixtures::build)
    }

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden)
                .map(|i| (i as f32 + 1.0) * 0.001)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn q4k_native_dispatch_trace_records_native_path() {
        // Regression for the LARQL_INSTRUMENT_MARKOV diagnosis: a Q4K-
        // only vindex (no gate_vectors.bin, no FP4) must NOT fall
        // through to `zero_features_dense`. After the num_features Q4K
        // fallback, the ladder reaches `interleaved_kquant:native`.
        let f = fx();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index).with_dispatch_trace();
        let x = input(1, f.weights.hidden_size);
        walk.forward(0, &x);
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].path, "interleaved_kquant:native");
    }

    #[test]
    fn q4k_native_no_longer_routes_to_zero_features_dense() {
        // The exact bug the session writeup pinned: `num_features == 0`
        // on a Q4K-only vindex used to short-circuit to the dense f32
        // `WeightFfn` fallback before any Q4K branch could run. Assert
        // that path is NEVER selected, across every layer.
        let f = fx();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index).with_dispatch_trace();
        let x = input(1, f.weights.hidden_size);
        for layer in 0..f.weights.num_layers {
            walk.forward(layer, &x);
        }
        let trace = walk.take_dispatch_trace();
        for entry in &trace {
            assert_ne!(
                entry.path, "zero_features_dense",
                "layer {} routed to zero_features_dense — the 100× FFN regression is back",
                entry.layer
            );
        }
    }

    #[test]
    fn q4k_native_output_matches_dequant_path() {
        // The dequant path (`walk_ffn_kquant_dequant`) is the ground-truth
        // baseline — it materialises full f32 gate/up/down then runs
        // dense matmul. Native does the same math via fused decode +
        // dot. Outputs must match within a tight tolerance (decode
        // arithmetic is deterministic; only floating-point order may
        // differ between row-parallel matvec and a single gemm).
        let f = fx();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index);
        let x = input(1, f.weights.hidden_size);

        let native = walk
            .walk_ffn_kquant_native(0, &x)
            .expect("native path must succeed on Q4K fixture");
        let dequant = walk
            .walk_ffn_kquant_dequant(0, &x)
            .expect("dequant path must succeed on Q4K fixture");

        assert_eq!(native.0.shape(), dequant.0.shape());
        for (a, b) in native.0.iter().zip(dequant.0.iter()) {
            let tol = (a.abs() + b.abs()).max(1.0) * 1e-4;
            assert!(
                (a - b).abs() < tol,
                "native vs dequant FFN output diverged: a={a} b={b} tol={tol}"
            );
        }
    }

    #[test]
    fn q4k_native_output_finite() {
        // Smoke test: the FFN block on the synthetic fixture produces
        // finite values for a single-token input. The fixture uses
        // small-scale weights (rand_mat_seeded with scale=0.05) so the
        // activations stay well-bounded — any NaN/Inf here indicates
        // a kernel-level regression in `kquant_matmul_transb` or
        // `gelu_tanh_gate_up`.
        let f = fx();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index);
        let x = input(1, f.weights.hidden_size);
        let (out, _) = walk
            .walk_ffn_kquant_native(0, &x)
            .expect("native path must succeed on Q4K fixture");
        assert!(out.iter().all(|v| v.is_finite()), "FFN output has NaN/Inf");
    }

    #[test]
    fn q4k_native_handles_multi_token_input() {
        // Prefill-style multi-token input. WalkFfn::forward operates on
        // `[seq_len, hidden]` — the native path must propagate seq_len
        // through `kquant_matmul_transb` correctly. Without this test a
        // regression that flattens to seq=1 would only surface on
        // prefill-heavy workloads.
        let f = fx();
        let walk = WalkFfn::new_unlimited(&f.weights, &f.index);
        let x = input(3, f.weights.hidden_size);
        let (out, _) = walk
            .walk_ffn_kquant_native(0, &x)
            .expect("native path must succeed for seq_len=3");
        assert_eq!(out.shape(), &[3, f.weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q4k_native_returns_none_without_q4k_bytes() {
        // No Q4K bytes installed → `interleaved_kquant_layer_data` is
        // None → native path returns None and caller falls through.
        // Uses the non-Q4K test fixture from `test_utils::TestFixtures`,
        // which has only safetensors weights.
        let fx = crate::test_utils::TestFixtures::build();
        let walk = WalkFfn::new_unlimited(&fx.weights, &fx.index);
        let x = input(1, fx.weights.hidden_size);
        assert!(walk.walk_ffn_kquant_native(0, &x).is_none());
    }
}
