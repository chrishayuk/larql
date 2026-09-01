//! Direct MXFP4 execution on Metal — K1 of the fused-kernel ladder.
//!
//! The API here is deliberately shaped for the execution model the later rungs
//! need, so that K1 does not bake in today's assumptions:
//!
//! - `layout` is explicit, because the checkpoint stores `weight_packed` and
//!   `weight_scale` as *separate* tensors while a future vindex sidecar may
//!   interleave them per row. Callers should never have to guess.
//! - `t` is a parameter from the start even though K1 only implements `t == 1`,
//!   so the small-block rung (K2) is a kernel swap rather than a signature
//!   change and no caller has to be rewritten.
//! - the grouped-expert entry point (K3) takes an assignment list rather than
//!   padded dense batches, because ragged position sets are what real routing
//!   produces.
//!
//! Ladder state: **K1 implemented** (`t == 1`). K2 (small block, `d(T) -> 1`),
//! K3 (grouped experts, `r(T) -> u(T)`), K4 (fused SiTU gate/up) and K5 (dense
//! KDA projections) are not.

use crate::MetalBackend;

/// How a caller's MXFP4 payload is arranged in memory.
///
/// Named rather than assumed: the released checkpoint keeps packed codes and
/// e8m0 scales in two separate tensors, but an interleaved per-row layout is a
/// live option for the serving sidecar and would change the kernel's addressing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp4Layout {
    /// `packed[M, K/32, 16]` and `scales[M, K/32]` as two separate buffers —
    /// the checkpoint's own arrangement.
    SeparateTensors,
    /// Packed codes and their scale contiguous per group. Not yet implemented;
    /// present so callers can express intent and get a clear error.
    InterleavedPerGroup,
}

/// Error cases the caller can act on, rather than a silent wrong answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mxfp4Error {
    /// `k` is not a whole number of 32-element groups.
    KNotGroupAligned { k: usize },
    /// Buffer smaller than the shape requires.
    ShortBuffer {
        what: &'static str,
        got: usize,
        need: usize,
    },
    /// The command buffer carrying this call finished in a state other
    /// than `Completed`; its output is not read back.
    CommandBufferFailed { site: &'static str, detail: String },
    /// A ladder rung that does not exist yet.
    Unimplemented(&'static str),
}

impl std::fmt::Display for Mxfp4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandBufferFailed { site, detail } => {
                write!(f, "command buffer at {site} did not complete: {detail}")
            }
            Self::KNotGroupAligned { k } => {
                write!(f, "MXFP4: K={k} is not a multiple of the 32-element group")
            }
            Self::ShortBuffer { what, got, need } => {
                write!(f, "MXFP4: {what} too short: {got} bytes < {need} required")
            }
            Self::Unimplemented(what) => write!(f, "MXFP4: {what} not implemented yet"),
        }
    }
}

impl std::error::Error for Mxfp4Error {}

/// Elements per e8m0 scale group.
pub const GROUP_ELEMS: usize = 32;
/// Packed bytes per group (32 codes at two per byte).
pub const GROUP_BYTES: usize = 16;

impl MetalBackend {
    /// `Y[m, t] = W_mxfp4[m, k] · X[k, t]`, consuming packed weights directly.
    ///
    /// No f32 weight buffer is allocated at any point: nibbles are unpacked and
    /// scaled in registers inside the kernel. Accumulation order differs from
    /// the CPU reference (lane-parallel then `simd_sum`, versus left-to-right),
    /// so results agree to a bounded error rather than bit-exactly.
    ///
    /// K1 implements `t == 1`. Larger `t` returns [`Mxfp4Error::Unimplemented`]
    /// rather than silently looping, because a loop here would be exactly the
    /// token-loop non-reuse that this ladder exists to remove — measured at
    /// `d(T) = r(T) = T`.
    // Eight arguments is the kernel's actual signature: two weight planes
    // (packed, scales), the activation, three shape scalars, and the layout.
    // Bundling them into a params struct would hide the shape contract that
    // the guards below check one by one.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_matmul_small_block(
        &self,
        packed: &[u8],
        scales: &[u8],
        x: &[f32],
        m: usize,
        k: usize,
        t: usize,
        layout: Mxfp4Layout,
    ) -> Result<Vec<f32>, Mxfp4Error> {
        if layout != Mxfp4Layout::SeparateTensors {
            return Err(Mxfp4Error::Unimplemented("interleaved-per-group layout"));
        }
        if t != 1 {
            return Err(Mxfp4Error::Unimplemented(
                "block width t > 1 (K2); refusing to emulate it with a token loop",
            ));
        }
        if !k.is_multiple_of(GROUP_ELEMS) {
            return Err(Mxfp4Error::KNotGroupAligned { k });
        }

        let groups = k / GROUP_ELEMS;
        let need_packed = m * groups * GROUP_BYTES;
        let need_scales = m * groups;
        if packed.len() < need_packed {
            return Err(Mxfp4Error::ShortBuffer {
                what: "packed",
                got: packed.len(),
                need: need_packed,
            });
        }
        if scales.len() < need_scales {
            return Err(Mxfp4Error::ShortBuffer {
                what: "scales",
                got: scales.len(),
                need: need_scales,
            });
        }
        if x.len() < k {
            return Err(Mxfp4Error::ShortBuffer {
                what: "x",
                got: x.len(),
                need: k,
            });
        }

        let buf_p = self.bufs.get_bytes(packed);
        let buf_s = self.bufs.get_bytes(scales);
        let buf_x = self.bufs.transient_from_f32(&x[..k]);
        let buf_out = self.bufs.output((m * 4) as u64);
        let m_u32 = m as u32;
        let k_u32 = k as u32;

        let handle = &self.quant.mxfp4_matvec_pipeline;
        let num_tgs = (m as u64).div_ceil(handle.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&handle.state);
        enc.set_buffer(0, Some(&buf_p), 0);
        enc.set_buffer(1, Some(&buf_s), 0);
        enc.set_buffer(2, Some(&buf_x), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.set_bytes(4, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(handle.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/mxfp4.rs:162",
        )
        .map_err(|detail| Mxfp4Error::CommandBufferFailed {
            site: "crates/larql-compute-metal/src/trait_impl/mxfp4.rs:162",
            detail,
        })?;

        Ok(crate::buffers::read_buffer_f32(&buf_out, m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::quant::mxfp4::dequantize_expert;

    /// The K3 expert down-projection shape, small enough to run in a unit test.
    const K: usize = 256;
    const M: usize = 9; // deliberately not a multiple of ROWS_PER_TG

    fn synth(m: usize, k: usize, seed: usize) -> (Vec<u8>, Vec<u8>) {
        let groups = k / GROUP_ELEMS;
        let packed = (0..m * groups * GROUP_BYTES)
            .map(|i| ((i * 37 + seed * 11) % 256) as u8)
            .collect();
        // Keep exponents near 127 so products stay in a comparable range, and
        // never emit 0 or 255 here — those get their own adversarial test.
        let scales = (0..m * groups).map(|i| (120 + (i % 15)) as u8).collect();
        (packed, scales)
    }

    fn reference(packed: &[u8], scales: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let w = dequantize_expert(packed, scales, m, k / GROUP_ELEMS).expect("reference dequant");
        (0..m)
            .map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum())
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    fn run(packed: &[u8], scales: &[u8], x: &[f32], m: usize, k: usize) -> Option<Vec<f32>> {
        let be = MetalBackend::new()?;
        Some(
            be.mxfp4_matmul_small_block(packed, scales, x, m, k, 1, Mxfp4Layout::SeparateTensors)
                .expect("dispatch"),
        )
    }

    #[test]
    fn matches_the_bulk_dequant_reference() {
        let (p, s) = synth(M, K, 1);
        let x: Vec<f32> = (0..K).map(|i| ((i % 251) as f32 - 125.0) * 0.01).collect();
        let Some(got) = run(&p, &s, &x, M, K) else {
            return;
        };
        let want = reference(&p, &s, &x, M, K);

        assert!(
            cosine(&got, &want) > 0.999_999,
            "cosine {}",
            cosine(&got, &want)
        );
        for (g, w) in got.iter().zip(&want) {
            let rel = (g - w).abs() / w.abs().max(1e-3);
            assert!(rel < 1e-4, "got {g} want {w} (rel {rel})");
        }
    }

    #[test]
    fn handles_rows_not_a_multiple_of_the_tile() {
        // M=9 with ROWS_PER_TG=4 leaves a partial final threadgroup; the guard
        // must not drop rows or write past the output.
        let (p, s) = synth(M, K, 2);
        let x: Vec<f32> = (0..K).map(|i| ((i % 97) as f32 - 48.0) * 0.02).collect();
        let Some(got) = run(&p, &s, &x, M, K) else {
            return;
        };
        assert_eq!(got.len(), M);
        let want = reference(&p, &s, &x, M, K);
        assert!(cosine(&got, &want) > 0.999_999);
    }

    #[test]
    fn reproduces_the_zero_scale_sentinel() {
        // e8m0 byte 0 means 0.0, not 2^-127.
        let groups = K / GROUP_ELEMS;
        let (p, mut s) = synth(M, K, 3);
        s[..groups].fill(0); // whole first row zeroed
        let x: Vec<f32> = (0..K).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
        let Some(got) = run(&p, &s, &x, M, K) else {
            return;
        };
        assert_eq!(got[0], 0.0, "row with all-zero scales must be exactly 0");
        let want = reference(&p, &s, &x, M, K);
        assert!(cosine(&got[1..], &want[1..]) > 0.999_999);
    }

    #[test]
    fn reproduces_the_nan_scale_sentinel() {
        // The reference returns NaN for byte 255; a raw bitcast would give
        // +inf. Diverging here would be invisible on real checkpoints (no
        // tensor contains 255) and wrong on adversarial input.
        let (p, mut s) = synth(M, K, 4);
        s[0] = 255;
        let x: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let Some(got) = run(&p, &s, &x, M, K) else {
            return;
        };
        assert!(
            got[0].is_nan(),
            "255 scale must propagate NaN, got {}",
            got[0]
        );
    }

    #[test]
    fn handles_extreme_codes_and_signs() {
        // 0x0F/0xF0 nibbles are -6.0; 0x88 is -0.0. Exercise the LUT tails.
        let groups = K / GROUP_ELEMS;
        let mut p = vec![0u8; M * groups * GROUP_BYTES];
        for (i, b) in p.iter_mut().enumerate() {
            *b = match i % 4 {
                0 => 0xFF,
                1 => 0x0F,
                2 => 0x88,
                _ => 0x70,
            };
        }
        let s: Vec<u8> = (0..M * groups).map(|i| (125 + (i % 5)) as u8).collect();
        let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.3).collect();
        let Some(got) = run(&p, &s, &x, M, K) else {
            return;
        };
        let want = reference(&p, &s, &x, M, K);
        assert!(
            cosine(&got, &want) > 0.999_99,
            "cosine {}",
            cosine(&got, &want)
        );
    }

    #[test]
    fn is_deterministic_across_repeats() {
        let (p, s) = synth(M, K, 5);
        let x: Vec<f32> = (0..K).map(|i| ((i % 251) as f32 - 125.0) * 0.01).collect();
        let Some(first) = run(&p, &s, &x, M, K) else {
            return;
        };
        for _ in 0..4 {
            assert_eq!(run(&p, &s, &x, M, K).unwrap(), first);
        }
    }

    #[test]
    fn rejects_a_token_loop_instead_of_emulating_one() {
        let (p, s) = synth(M, K, 6);
        let x: Vec<f32> = vec![0.5; K * 4];
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let err = be
            .mxfp4_matmul_small_block(&p, &s, &x, M, K, 4, Mxfp4Layout::SeparateTensors)
            .unwrap_err();
        assert!(matches!(err, Mxfp4Error::Unimplemented(_)));
    }

    #[test]
    fn rejects_unaligned_k_and_short_buffers() {
        let (p, s) = synth(M, K, 7);
        let x = vec![0.1f32; K];
        let Some(be) = MetalBackend::new() else {
            return;
        };
        let l = Mxfp4Layout::SeparateTensors;
        assert!(matches!(
            be.mxfp4_matmul_small_block(&p, &s, &x, M, K + 1, 1, l),
            Err(Mxfp4Error::KNotGroupAligned { .. })
        ));
        assert!(matches!(
            be.mxfp4_matmul_small_block(&p[..16], &s, &x, M, K, 1, l),
            Err(Mxfp4Error::ShortBuffer { what: "packed", .. })
        ));
        assert!(matches!(
            be.mxfp4_matmul_small_block(&p, &s[..2], &x, M, K, 1, l),
            Err(Mxfp4Error::ShortBuffer { what: "scales", .. })
        ));
    }

    #[test]
    fn rejects_an_x_shorter_than_one_reduction() {
        // x is indexed to K regardless of M, so a short activation would read
        // past the buffer rather than produce a small answer.
        let (p, s) = synth(M, K, 8);
        let Some(be) = MetalBackend::new() else {
            return;
        };
        assert!(matches!(
            be.mxfp4_matmul_small_block(
                &p,
                &s,
                &vec![0.1f32; K - 1],
                M,
                K,
                1,
                Mxfp4Layout::SeparateTensors
            ),
            Err(Mxfp4Error::ShortBuffer { what: "x", .. })
        ));
    }

    #[test]
    fn rejects_the_interleaved_layout_rather_than_misreading_it() {
        // Interleaved-per-group packs scales beside codewords, so decoding it
        // with the separate-tensor addressing yields plausible numbers from the
        // wrong bytes. It has to be refused, not attempted.
        let (p, s) = synth(M, K, 9);
        let x = vec![0.25f32; K];
        let Some(be) = MetalBackend::new() else {
            return;
        };
        assert!(matches!(
            be.mxfp4_matmul_small_block(&p, &s, &x, M, K, 1, Mxfp4Layout::InterleavedPerGroup),
            Err(Mxfp4Error::Unimplemented(_))
        ));
    }

    #[test]
    fn every_error_variant_renders_its_own_diagnostic() {
        // These strings are what a caller sees when a shape contract fails, so
        // each must name the quantity that was wrong.
        let k_err = Mxfp4Error::KNotGroupAligned { k: 257 }.to_string();
        assert!(k_err.contains("257"), "{k_err}");

        let short = Mxfp4Error::ShortBuffer {
            what: "packed",
            got: 16,
            need: 4608,
        }
        .to_string();
        assert!(short.contains("packed") && short.contains("16") && short.contains("4608"));

        let unimpl = Mxfp4Error::Unimplemented("block width t > 1").to_string();
        assert!(unimpl.contains("block width t > 1"));

        // The Error impl is what lets these cross an api boundary as `dyn Error`.
        let boxed: Box<dyn std::error::Error> = Box::new(Mxfp4Error::KNotGroupAligned { k: 3 });
        assert!(boxed.to_string().contains('3'));
    }
}
