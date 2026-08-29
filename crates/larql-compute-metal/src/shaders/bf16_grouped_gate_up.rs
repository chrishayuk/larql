//! Fused grouped gate+up — one dispatch, one traversal of the input.
//!
//! Rung 4 of the Kimi Metal ladder. Rung 3 removed the submission cost
//! (three command buffers to one: 1.084 → 0.707 ms wall, GPU-busy flat
//! at ~0.42 ms), which is what makes this measurable at all — a fusion
//! win of a few percent would have been invisible underneath a 0.4 ms
//! orchestration cost.
//!
//! ## Two kernels, because there are two mechanisms
//!
//! `bf16_grouped_gate_up` computes both projections in one dispatch and
//! writes them to separate outputs. Against two separate grouped
//! dispatches it changes exactly one thing: each simdgroup loads `X[k]`
//! once and uses it for both accumulators, instead of two simdgroups
//! loading it separately.
//!
//! `bf16_grouped_gate_up_silu` goes further and applies
//! `silu(gate) * up` in register, writing only the product. That removes
//! the standalone activation dispatch and two intermediate streams —
//! one write instead of two, and no read-back at all.
//!
//! Emitting both is the whole point: `gate_up` against separate dispatches
//! prices the shared input traversal, and `gate_up_silu` against
//! `gate_up` prices the intermediate traffic. One fused kernel would
//! have conflated them.
//!
//! ## MEASURED: fusion is a reproducible 8-9% REGRESSION
//!
//! Kept because it is the evidence, not because it is the fast path.
//! **`BlockLowering::Separate` remains the default and should stay
//! there** at these shapes.
//!
//! At Kimi's layer-1 geometry, one command buffer throughout, arms
//! interleaved, ramp 0.99-1.01, three runs:
//!
//! ```text
//! Separate                4 disp  2304 TGs  gpu 0.418-0.422 ms  302-305 GB/s
//! FusedGateUp(Rows8)      3 disp  1152 TGs  gpu 0.457-0.462 ms  276-279 GB/s  0.907-0.917x
//! FusedGateUp(Rows4)      3 disp  2304 TGs  gpu 0.457-0.461 ms  276-279 GB/s  0.913-0.916x
//! FusedGateUpAct(Rows8)   2 disp  1152 TGs  gpu 0.457-0.460 ms  277-279 GB/s  0.913-0.918x
//! FusedGateUpAct(Rows4)   2 disp  2304 TGs  gpu 0.455-0.462 ms  276-280 GB/s  0.911-0.919x
//! ```
//!
//! Every arm is bit-identical to `Separate`, so this is a scheduling
//! result and not a numerics one.
//!
//! **Why there was nothing to win.** The prediction written here before
//! measuring was that fusion could not help much because the weight
//! traffic is unchanged and the unfused kernel already runs near
//! roofline. Instrumenting the traffic confirmed the premise and then
//! some: intermediates are **0.173% of bytes moved**, and the activated
//! fusion cuts them to 0.058%. There is no traffic mechanism here — the
//! saving is under two tenths of one percent of what the block reads.
//!
//! **Why it is actively worse.** `Rows4` is the control that answers
//! this: fusing at the same tiling also halves the launch, so `Rows4`
//! restores the threadgroup count `Separate` issues. The regression is
//! unchanged at both tilings, which rules out launch size and leaves the
//! kernel itself — a simdgroup now walks two weight rows concurrently
//! instead of one, doubling live accumulators and the streams in flight
//! per thread. One row per simdgroup is evidently what this memory
//! system prefers, and dropping two dispatches does not pay for
//! disturbing it.
//!
//! **This is a shape-specific result, not a law.** K3's expert branch is
//! `[3584, 3072]` against Kimi's `[1024, 2304]`; the arms are kept so
//! the question can be re-asked there rather than re-implemented.
//!
//! ## Bit-exactness
//!
//! Each accumulator walks its row exactly as [`bf16_grouped_experts`]
//! does — same lane stride, same four-way unroll, same `simd_sum`, and
//! `fma` against the same `X[k]`. Interleaving two independent
//! accumulations does not reassociate either. The activation is the same
//! `(g / (1 + exp(-g))) * up` the crate's `geglu_silu` evaluates, now in
//! register instead of through a buffer, which changes no value. So all
//! three lowerings must agree **exactly**, and the gate asserts that
//! rather than a tolerance.
//!
//! ## Two tilings, and why the second exists
//!
//! `r=8` matches the unfused default, so the fused-vs-separate
//! comparison differs in fusion alone. But that comparison changes the
//! LAUNCH as a side effect: separate gate and up issue `2 x 1152`
//! threadgroups for the same work a fused dispatch does in `1152`, each
//! doing twice as much. Rung 2's sweep found tiling flat for a
//! one-row-per-simdgroup kernel, which does not license the same claim
//! about a kernel that doubled its per-thread work and register
//! pressure.
//!
//! So `r=4` is emitted too: at `[1024, 2304]` it launches `256 x 9 =
//! 2304` threadgroups, exactly what separate gate+up issue. If a fused
//! regression survives at matched threadgroup count, it is register
//! pressure or the doubled per-simdgroup stream count, not the launch.
//!
//! [`bf16_grouped_experts`]: super::bf16_grouped_experts

/// Row tilings emitted. `8` matches the unfused default; `4` matches
/// the THREADGROUP COUNT that separate gate and up issue between them,
/// which is the control that tells a launch effect from a register one.
pub const ROWS_PER_TG_VARIANTS: [u64; 2] = [8, 4];

/// The default tiling, matched to
/// [`super::bf16_grouped_experts::ROWS_PER_TG`].
pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Metal kernel function names for a given tiling.
pub fn gate_up_name(rows: u64) -> String {
    format!("bf16_grouped_gate_up_r{rows}")
}
pub fn gate_up_silu_name(rows: u64) -> String {
    format!("bf16_grouped_gate_up_silu_r{rows}")
}

/// The shared prologue and accumulation, parameterised only by what the
/// kernel does with the two sums.
fn kernel_source(name: &str, rows: u64, outputs: &str, epilogue: &str) -> String {
    format!(
        r#"
kernel void {name}(
    device const ushort* WG      [[buffer(0)]],  // gate bank, bf16 codes
    device const uint*   OG      [[buffer(1)]],  // [n_sel] BYTE offset per slot
    device const ushort* WU      [[buffer(2)]],  // up bank, bf16 codes
    device const uint*   OU      [[buffer(3)]],  // [n_sel] BYTE offset per slot
    device const float*  X       [[buffer(4)]],  // [K], shared by every slot
{outputs}
    constant uint&       N       [[buffer(7)]],
    constant uint&       K       [[buffer(8)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{{
    constexpr uint BF16GU_ROWS_PER_TG = {rows};
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * BF16GU_ROWS_PER_TG + sg_id;
    if (row >= N) {{ return; }}

    // `>> 1` because the offset tables are in bytes and these pointers
    // are ushort — the same convention every grouped kernel here uses.
    device const ushort* g_row = WG + (OG[slot] >> 1u) + (ulong)row * K;
    device const ushort* u_row = WU + (OU[slot] >> 1u) + (ulong)row * K;

    // Two independent accumulations sharing one traversal of X. Each is
    // element-for-element what `bf16_grouped_experts` computes for the
    // same row, so neither is reassociated by the interleaving.
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {{
        float x0 = X[k         ];
        float x1 = X[k + 32    ];
        float x2 = X[k + 64    ];
        float x3 = X[k + 96    ];
        g0 = fma(decode_bf16_metal(g_row[k         ]), x0, g0);
        g1 = fma(decode_bf16_metal(g_row[k + 32    ]), x1, g1);
        g2 = fma(decode_bf16_metal(g_row[k + 64    ]), x2, g2);
        g3 = fma(decode_bf16_metal(g_row[k + 96    ]), x3, g3);
        u0 = fma(decode_bf16_metal(u_row[k         ]), x0, u0);
        u1 = fma(decode_bf16_metal(u_row[k + 32    ]), x1, u1);
        u2 = fma(decode_bf16_metal(u_row[k + 64    ]), x2, u2);
        u3 = fma(decode_bf16_metal(u_row[k + 96    ]), x3, u3);
    }}
    float gacc = (g0 + g1) + (g2 + g3);
    float uacc = (u0 + u1) + (u2 + u3);
    for (; k < K; k += 32) {{
        float xv = X[k];
        gacc = fma(decode_bf16_metal(g_row[k]), xv, gacc);
        uacc = fma(decode_bf16_metal(u_row[k]), xv, uacc);
    }}

    gacc = simd_sum(gacc);
    uacc = simd_sum(uacc);
    if (lane == 0) {{
{epilogue}
    }}
}}
"#,
        rows = rows,
    )
}

/// Every fused variant at every tiling, for the shader library.
pub fn shader() -> String {
    let separate_outputs = "    device float*        gate_out [[buffer(5)]],  // [n_sel, N]\n\
                            \x20   device float*        up_out   [[buffer(6)]],  // [n_sel, N]";
    let product_outputs = "    device float*        out      [[buffer(5)]],  // [n_sel, N]\n\
                           \x20   device float*        unused   [[buffer(6)]],  // bound, never written";
    ROWS_PER_TG_VARIANTS
        .iter()
        .flat_map(|&rows| {
            [
                kernel_source(
                    &gate_up_name(rows),
                    rows,
                    separate_outputs,
                    "        gate_out[slot * N + row] = gacc;\n        \
                     up_out[slot * N + row] = uacc;",
                ),
                kernel_source(
                    &gate_up_silu_name(rows),
                    rows,
                    product_outputs,
                    "        // Same expression as the crate's `geglu_silu`, now in\n        \
                     // register: silu(gate) * up.\n        \
                     out[slot * N + row] = (gacc / (1.0f + exp(-gacc))) * uacc;",
                ),
            ]
        })
        .collect()
}

/// One `TiledKernel` marker per emitted variant, so binding sites read
/// names from constants. Pinned against the generators by
/// `marker_names_match_the_generators`.
macro_rules! fused_variant {
    ($ident:ident, $rows:literal, $name:literal) => {
        pub struct $ident;
        impl crate::kernels::TiledKernel for $ident {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = $rows;
            const THREADS_PER_TG: u64 = $rows * 32;
        }
    };
}

fused_variant!(GateUpKernelR8, 8, "bf16_grouped_gate_up_r8");
fused_variant!(GateUpKernelR4, 4, "bf16_grouped_gate_up_r4");
fused_variant!(GateUpSiluKernelR8, 8, "bf16_grouped_gate_up_silu_r8");
fused_variant!(GateUpSiluKernelR4, 4, "bf16_grouped_gate_up_silu_r4");

#[cfg(test)]
mod tests {
    use super::*;

    /// The two variants must share one accumulation body — that is what
    /// lets the pair attribute two mechanisms separately instead of
    /// comparing two different kernels.
    #[test]
    fn both_variants_share_one_accumulation_body() {
        let src = shader();
        let bodies: Vec<&str> = src.split("kernel void ").skip(1).collect();
        assert_eq!(
            bodies.len(),
            2 * ROWS_PER_TG_VARIANTS.len(),
            "two variants per tiling expected"
        );
        let core = |b: &str| {
            let start = b.find("float g0 = 0.0f").expect("accumulator prologue");
            let end = b.find("uacc = simd_sum(uacc);").expect("reduction");
            b[start..end].to_string()
        };
        let first = core(bodies[0]);
        for b in &bodies[1..] {
            assert_eq!(core(b), first, "a variant diverged from the shared body");
        }
    }

    /// The accumulation must walk the row exactly as the unfused grouped
    /// kernel does, or "fusion changes no value" is not true.
    #[test]
    fn the_traversal_matches_the_unfused_grouped_kernel() {
        let fused = shader();
        for stride in ["k = lane;", "k + 3 * 32 < K", "k += 4 * 32", "k += 32"] {
            assert!(fused.contains(stride), "fused kernel missing `{stride}`");
            assert!(
                super::super::bf16_grouped_experts::shader().contains(stride),
                "unfused kernel missing `{stride}`"
            );
        }
        // Same pairwise reduction tree, written for each accumulator.
        assert!(fused.contains("float gacc = (g0 + g1) + (g2 + g3);"));
        assert!(fused.contains("float uacc = (u0 + u1) + (u2 + u3);"));
    }

    /// Geometry matches the unfused grouped kernel so a fused-vs-separate
    /// comparison differs in fusion alone.
    #[test]
    fn geometry_matches_the_grouped_default() {
        assert_eq!(ROWS_PER_TG, super::super::bf16_grouped_experts::ROWS_PER_TG);
        assert_eq!(
            THREADS_PER_TG,
            super::super::bf16_grouped_experts::THREADS_PER_TG
        );
        for rows in ROWS_PER_TG_VARIANTS {
            for name in [gate_up_name(rows), gate_up_silu_name(rows)] {
                assert!(shader().contains(&format!("kernel void {name}(")));
            }
        }
    }

    /// The markers and the generators must name the same functions, and
    /// `r=4` must be the tiling that matches what separate gate and up
    /// issue between them — that equality is the control's whole point.
    #[test]
    fn marker_names_match_the_generators() {
        use crate::kernels::TiledKernel;
        assert_eq!(GateUpKernelR8::KERNEL_NAME, gate_up_name(8));
        assert_eq!(GateUpKernelR4::KERNEL_NAME, gate_up_name(4));
        assert_eq!(GateUpSiluKernelR8::KERNEL_NAME, gate_up_silu_name(8));
        assert_eq!(GateUpSiluKernelR4::KERNEL_NAME, gate_up_silu_name(4));
        for k in [
            GateUpKernelR4::ROWS_PER_TG * 2,
            GateUpSiluKernelR4::ROWS_PER_TG * 2,
        ] {
            assert_eq!(
                k,
                super::super::bf16_grouped_experts::ROWS_PER_TG,
                "one fused threadgroup at r4 must cover the rows two \
                 unfused threadgroups do between them"
            );
        }
    }
}
