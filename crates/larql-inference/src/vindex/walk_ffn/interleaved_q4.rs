//! Q4_0 interleaved walk. C kernel with `vdotq_s32` for gate/up, scalar
//! kernel for down. Reads ~44 MB per layer (vs 315 MB for f32
//! interleaved) — 7× less data to page in, same BLAS speed warm.
//!
//! Batched GPU path (when the backend implements
//! `q4_matvec_pair_batch` — probed by calling it, since
//! `supports_quant` can't distinguish per-row kernels from the batch
//! API): one GPU submission for gate+up across all seq positions,
//! followed by one vecmat per position for down. C kernel path is the
//! CPU fallback, including for backends that decline the batch call.

use ndarray::Array2;

use super::WalkFfn;
use larql_vindex::{FFN_COMPONENTS_PER_LAYER, FFN_DOWN, FFN_GATE, FFN_UP};

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_q4_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

        let q4_mmap = self.index.interleaved_q4_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4_bytes_per_matrix = larql_compute::QuantFormat::Q4_0
            .packed_matrix_bytes(intermediate, hidden)
            .expect("Q4_0 interleaved FFN format must have packed geometry");
        let q4_bytes_per_layer = q4_bytes_per_matrix * FFN_COMPONENTS_PER_LAYER;
        let layer_start = layer * q4_bytes_per_layer;

        // Component slices in wire order (gate, up, down).
        let component = |c: usize| {
            &q4_mmap
                [layer_start + c * q4_bytes_per_matrix..layer_start + (c + 1) * q4_bytes_per_matrix]
        };
        let gate_q4 = component(FFN_GATE);
        let up_q4 = component(FFN_UP);
        let down_q4 = component(FFN_DOWN);

        self.index.prefetch_interleaved_q4_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = arch.gate_up_is_gelu_tanh();

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Batch capability is probed by CALLING `q4_matvec_pair_batch`,
        // never via `supports_quant`: CpuBackend advertises Q4-family
        // matvec kernels but leaves the batch API at its trait default
        // (`None`). A `None` here falls through to the C-kernel loop —
        // the pre-fix code filtered on `supports_quant(Q4_K)` and
        // unwrapped, panicking on any non-Metal backend.
        let batch = self.backend.and_then(|be| {
            let x_flat = x.as_slice()?;
            be.q4_matvec_pair_batch(gate_q4, up_q4, x_flat, seq_len, intermediate, hidden)
                .map(|pair| (be, pair))
        });

        if let Some((be, (all_gate, all_up))) = batch {
            // Metal: ONE GPU submission for all gate+up across ALL seq positions
            let mut all_activation: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = all_gate[s][i];
                    let u = all_up[s][i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }
                all_activation.push(activation);
            }

            for (s, activation_row) in all_activation.iter().enumerate().take(seq_len) {
                // A backend that batches gate/up should also vecmat down;
                // if it declines, finish the row on the C kernel rather
                // than losing the layer (or panicking) mid-forward.
                let down_result = be
                    .q4_vecmat(activation_row, down_q4, intermediate, hidden)
                    .unwrap_or_else(|| {
                        q4_vecmat::dispatch(activation_row, down_q4, intermediate, hidden)
                    });
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:metal");
        } else {
            for s in 0..seq_len {
                let x_row = x.row(s);
                let x_slice = x_row.as_slice().unwrap();

                let gate_scores = q4_matvec::dispatch(gate_q4, x_slice, intermediate, hidden);
                let up_scores = q4_matvec::dispatch(up_q4, x_slice, intermediate, hidden);

                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = gate_scores[i];
                    let u = up_scores[i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }

                let down_result = q4_vecmat::dispatch(&activation, down_q4, intermediate, hidden);
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:cpu");
        }

        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }
}

#[cfg(test)]
mod tests {
    //! First tests for this file (2026-07-30 review, finding H2): the
    //! pre-fix path filtered the GPU branch on `supports_quant(Q4_K)` —
    //! true for `CpuBackend`, which nevertheless leaves
    //! `q4_matvec_pair_batch` at its trait default (`None`) — then
    //! `.unwrap()`ed the batch call: panic on the first Q4_0 layer for
    //! any non-Metal backend.
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    use larql_compute::CpuBackend;
    use larql_models::test_fixtures::arc_mmap_from_bytes;
    use larql_vindex::QuantizedFfnAccess;
    use ndarray::Array2;

    use crate::ffn::FfnBackend;
    use crate::test_utils::Q4KTestFixtures;
    use crate::vindex::WalkFfn;

    /// Feature count for the synthetic Q4_0 layer. Small keeps the test
    /// fast; `hidden` (from the fixture) satisfies the Q4_0 kernel's
    /// 32-element block requirement.
    const INTERMEDIATE: usize = 8;

    /// One-layer Q4_0-only vindex: raw `[gate|up|down]` slab, no
    /// manifest (the format has none), gate matrix sized so
    /// `num_features(0) == INTERMEDIATE`.
    fn q4_0_index(hidden: usize) -> larql_vindex::VectorIndex {
        let n = INTERMEDIATE * hidden;
        let mat = |seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) % 13) as f32 * 0.01 - 0.06)
                .collect::<Vec<f32>>()
        };
        let mut payload = Vec::new();
        for m in [mat(0), mat(5), mat(9)] {
            payload.extend_from_slice(&quantize_q4_0(&m));
        }
        let gate_vectors = vec![Some(Array2::<f32>::zeros((INTERMEDIATE, hidden)))];
        let down_meta = vec![None];
        let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, 1, hidden);
        let mmap = arc_mmap_from_bytes(&payload);
        std::sync::Arc::make_mut(&mut index.storage).set_interleaved_q4(mmap);
        index
    }

    fn input(hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (1, hidden),
            (0..hidden).map(|i| (i as f32 + 1.0) * 0.001).collect(),
        )
        .unwrap()
    }

    #[test]
    fn cpu_backend_runs_c_kernel_instead_of_panicking() {
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;

        let walk =
            WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu).with_dispatch_trace();
        let (out, activation) = walk
            .walk_ffn_q4_interleaved(0, &input(hidden))
            .expect("Q4_0 path must succeed with a CPU backend");

        assert_eq!(out.shape(), &[1, hidden]);
        assert_eq!(activation.shape(), &[1, INTERMEDIATE]);
        assert!(out.iter().all(|v| v.is_finite()));

        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(
            trace[0].path, "interleaved_q4:cpu",
            "CpuBackend must take the C-kernel branch, not the batch branch"
        );
    }

    #[test]
    fn cpu_backend_output_matches_backendless_c_kernel() {
        // A backend whose batch call declines must produce EXACTLY the
        // no-backend result — both run the same C kernel on the same
        // bytes, so this is equality, not tolerance.
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;
        let x = input(hidden);

        let with_backend = WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("backend run succeeds");
        let backendless = WalkFfn::new_unlimited(&f.weights, &index)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("backendless run succeeds");

        assert_eq!(with_backend.0, backendless.0);
        assert_eq!(with_backend.1, backendless.1);
    }

    /// Minimal backend that implements `q4_matvec_pair_batch` (and,
    /// when `vecmat` is set, `q4_vecmat`) by running the SAME CPU Q4_0
    /// kernels the C-kernel branch uses — so the batch ("Metal") branch
    /// must reproduce the CPU branch's output exactly, and the test can
    /// assert equality rather than shape.
    struct BatchQ4Backend {
        inner: CpuBackend,
        vecmat: bool,
    }

    impl larql_compute::MatMul for BatchQ4Backend {
        fn matmul(
            &self,
            a: ndarray::ArrayView2<f32>,
            b: ndarray::ArrayView2<f32>,
        ) -> ndarray::Array2<f32> {
            self.inner.matmul(a, b)
        }
        fn matmul_transb(
            &self,
            a: ndarray::ArrayView2<f32>,
            b: ndarray::ArrayView2<f32>,
        ) -> ndarray::Array2<f32> {
            self.inner.matmul_transb(a, b)
        }
    }

    impl larql_compute::QuantMatVec for BatchQ4Backend {
        fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
            self.inner.supports_quant(format)
        }
        fn q4_matvec_pair_batch(
            &self,
            gate_q4: &[u8],
            up_q4: &[u8],
            x_matrix: &[f32],
            seq_len: usize,
            num_rows: usize,
            hidden: usize,
        ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
            use larql_compute::cpu::ops::q4_matvec;
            let mut gates = Vec::with_capacity(seq_len);
            let mut ups = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let xs = &x_matrix[s * hidden..(s + 1) * hidden];
                gates.push(q4_matvec::dispatch(gate_q4, xs, num_rows, hidden));
                ups.push(q4_matvec::dispatch(up_q4, xs, num_rows, hidden));
            }
            Some((gates, ups))
        }
        fn q4_vecmat(
            &self,
            activation: &[f32],
            q4_data: &[u8],
            intermediate: usize,
            hidden: usize,
        ) -> Option<Vec<f32>> {
            use larql_compute::cpu::ops::q4_vecmat;
            self.vecmat
                .then(|| q4_vecmat::dispatch(activation, q4_data, intermediate, hidden))
        }
    }

    impl larql_compute::DecodeBackend for BatchQ4Backend {}

    impl larql_compute::ComputeBackend for BatchQ4Backend {
        fn name(&self) -> &str {
            "batch-q4-mock"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// The batch ("Metal") branch — gate/up in one submission, down via
    /// the backend's `q4_vecmat` — must equal the backendless C-kernel
    /// run bit-for-bit: the mock delegates to the same kernels.
    #[test]
    fn batch_backend_branch_equals_c_kernel_branch() {
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let backend = BatchQ4Backend {
            inner: CpuBackend,
            vecmat: true,
        };
        // Two positions so the per-position batch loops iterate.
        let x = Array2::from_shape_vec(
            (2, hidden),
            (0..2 * hidden).map(|i| (i as f32 + 1.0) * 0.001).collect(),
        )
        .unwrap();

        let walk =
            WalkFfn::new_unlimited_with_backend(&f.weights, &index, &backend).with_dispatch_trace();
        let batch = walk
            .walk_ffn_q4_interleaved(0, &x)
            .expect("batch branch runs");
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(
            trace[0].path, "interleaved_q4:metal",
            "a backend answering the batch probe must take the batch branch"
        );

        let cpu = WalkFfn::new_unlimited(&f.weights, &index)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("C-kernel branch runs");
        assert_eq!(batch.0, cpu.0, "output must be identical");
        assert_eq!(batch.1, cpu.1, "activation must be identical");
    }

    /// The mock is itself under this file's coverage — pin its
    /// delegation surface: math delegates to `CpuBackend`, and the
    /// `ComputeBackend` identity methods behave.
    #[test]
    fn batch_q4_mock_delegates_math_to_cpu_backend() {
        use larql_compute::{ComputeBackend, MatMul, QuantMatVec};
        let backend = BatchQ4Backend {
            inner: CpuBackend,
            vecmat: true,
        };
        assert_eq!(backend.name(), "batch-q4-mock");
        assert!(backend.as_any().is::<BatchQ4Backend>());
        assert_eq!(
            backend.supports_quant(larql_compute::QuantFormat::Q4_0),
            CpuBackend.supports_quant(larql_compute::QuantFormat::Q4_0)
        );
        let a = ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let b = ndarray::arr2(&[[5.0f32, 6.0], [7.0, 8.0]]);
        assert_eq!(
            backend.matmul(a.view(), b.view()),
            CpuBackend.matmul(a.view(), b.view())
        );
        assert_eq!(
            backend.matmul_transb(a.view(), b.view()),
            CpuBackend.matmul_transb(a.view(), b.view())
        );
    }

    /// SiLU-arch coverage of BOTH branches: the tinymodel Q4K weights
    /// (`make_test_q4k_weights_silu`) select the `g·σ(g)·u` arm. The
    /// activation must equal the hand-computed SiLU of the C-kernel
    /// gate/up scores, and the batch branch must equal the CPU branch.
    #[test]
    fn silu_arch_activation_matches_hand_computed_in_both_branches() {
        use larql_compute::cpu::ops::q4_matvec;
        use larql_models::test_fixtures::make_test_q4k_weights_silu;
        let weights = make_test_q4k_weights_silu();
        assert!(
            matches!(weights.arch.activation(), larql_models::Activation::Silu),
            "fixture must select the SiLU arm"
        );
        let hidden = weights.hidden_size;
        let index = q4_0_index(hidden);
        let x = input(hidden);

        // CPU branch.
        let cpu_walk = WalkFfn::new_unlimited(&weights, &index);
        let (out_cpu, act_cpu) = cpu_walk
            .walk_ffn_q4_interleaved(0, &x)
            .expect("C-kernel branch runs");

        // Hand-computed reference from the same Q4_0 kernels.
        let q4_bytes = larql_compute::QuantFormat::Q4_0
            .packed_matrix_bytes(INTERMEDIATE, hidden)
            .expect("geometry");
        let mmap = index.interleaved_q4_mmap_ref().expect("q4 slab");
        let gate_q4 = &mmap[..q4_bytes];
        let up_q4 = &mmap[q4_bytes..2 * q4_bytes];
        let x_slice = x.row(0).to_vec();
        let g = q4_matvec::dispatch(gate_q4, &x_slice, INTERMEDIATE, hidden);
        let u = q4_matvec::dispatch(up_q4, &x_slice, INTERMEDIATE, hidden);
        for i in 0..INTERMEDIATE {
            let expected = g[i] * crate::ffn::sigmoid(g[i]) * u[i];
            assert_eq!(
                act_cpu[[0, i]],
                expected,
                "feature {i}: SiLU activation must be bit-exact vs the same kernels"
            );
        }

        // Batch branch on the same weights — silu arm of the batch loop.
        let backend = BatchQ4Backend {
            inner: CpuBackend,
            vecmat: true,
        };
        let batch = WalkFfn::new_unlimited_with_backend(&weights, &index, &backend)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("batch branch runs");
        assert_eq!(batch.0, out_cpu);
        assert_eq!(batch.1, act_cpu);
    }

    /// A layer reporting zero features must decline (early guard) so the
    /// ladder can fall through instead of building 0-width matrices.
    #[test]
    fn declines_when_layer_has_zero_features() {
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        // Zero-row gate matrix → num_features(0) == 0, but a Q4_0 slab
        // is present so the mmap probe succeeds first.
        let gate_vectors = vec![Some(Array2::<f32>::zeros((0, hidden)))];
        let mut index = larql_vindex::VectorIndex::new(gate_vectors, vec![None], 1, hidden);
        let payload = quantize_q4_0(&vec![0.0f32; hidden]);
        std::sync::Arc::make_mut(&mut index.storage)
            .set_interleaved_q4(arc_mmap_from_bytes(&payload));
        assert_eq!(index.num_features(0), 0);
        let walk = WalkFfn::new_unlimited(&f.weights, &index);
        assert!(walk.walk_ffn_q4_interleaved(0, &input(hidden)).is_none());
    }

    /// A backend that batches gate/up but declines `q4_vecmat` must
    /// finish the down leg on the C kernel — same exact output, no
    /// panic (the pre-fix code unwrapped here).
    #[test]
    fn batch_backend_without_vecmat_finishes_down_on_c_kernel() {
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let backend = BatchQ4Backend {
            inner: CpuBackend,
            vecmat: false,
        };
        let x = input(hidden);

        let walk =
            WalkFfn::new_unlimited_with_backend(&f.weights, &index, &backend).with_dispatch_trace();
        let batch = walk
            .walk_ffn_q4_interleaved(0, &x)
            .expect("batch branch with declined vecmat still completes");
        assert_eq!(walk.take_dispatch_trace()[0].path, "interleaved_q4:metal");

        let cpu = WalkFfn::new_unlimited(&f.weights, &index)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("C-kernel branch runs");
        assert_eq!(batch.0, cpu.0);
        assert_eq!(batch.1, cpu.1);
    }

    #[test]
    fn routing_ladder_admits_cpu_backend_to_q4_0_path() {
        // The ladder gate (walk_ffn/mod.rs step 5) now checks the format
        // the data actually is (Q4_0). CpuBackend advertises Q4_0, so a
        // full forward routes here and completes on the C kernel.
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;

        let walk =
            WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu).with_dispatch_trace();
        let out = walk.forward(0, &input(hidden));
        assert!(out.iter().all(|v| v.is_finite()));

        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].path, "interleaved_q4:cpu");
    }
}
