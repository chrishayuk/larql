//! BF16 grouped-expert matvec — every selected expert in ONE dispatch.
//!
//! Rung 2 of the Kimi Metal ladder, and the same hypothesis
//! [`q6k_grouped_experts`](super::q6k_grouped_experts) was built to test,
//! at a different codec and a much skinnier shape.
//!
//! **The measurement that earned it.** Rung 1 put one BF16 expert FFN on
//! the GPU and found the arm was not bandwidth-bound at all: a gemv
//! reading 512 bytes cost 0.197 ms through the same entry point as a
//! real one costing 0.268 ms, so ~0.2 ms of every call was fixed
//! submission cost. Batching nine real expert GEMVs into one command
//! buffer — same kernel, same bytes, same arguments, only the submission
//! changed — won 4.68x and lifted achieved bandwidth from 20.0 to 93.5
//! GB/s. That removed the submission tax and left a second question:
//! 93.5 GB/s is still far off this machine's ~370 GB/s roofline, and one
//! Kimi expert projection at `[1024, 2304]` launches only `1024 / 8 =
//! 128` threadgroups. This kernel asks whether the residue is occupancy.
//!
//! A token selects eight experts plus a shared branch, so dispatched
//! together they supply `128 x 9 = 1152` threadgroups from the model's
//! own structure — the same argument `q6k_grouped_experts` makes for K3,
//! where the multiplier is 16.
//!
//! ## What this kernel changes, and what it deliberately does not
//!
//! Only the addressing. The reduction body is copied verbatim from
//! [`bf16_gemv`](super::bf16_gemv) — same lane stride, same four-way
//! unroll, same `simd_sum` tree — so outputs must agree with N separate
//! `bf16_gemv` dispatches **exactly**, not to a tolerance. Anything else
//! would confound an occupancy result with a numerics change.
//!
//! Specifically NOT done here: no fusion of gate/up/down, no activation
//! inside the kernel, no routing, no padding to inflate the threadgroup
//! count. Padding would raise apparent occupancy while adding useless
//! work; the win has to come from scheduling *real* independent experts
//! together.
//!
//! ## Dispatch
//!
//! Grid is 2-D: `(row_tiles, n_selected)`. Threadgroup `(tx, ty)` handles
//! row tile `tx` of the expert in slot `ty`, reads its payload base from
//! `offsets[ty]`, and writes `out[ty * N + row]`. Expert identity travels
//! in the offset table, never in a row's position — so a slot may point
//! anywhere in the bank, two slots may name the same expert, and the
//! caller combines the `[n_selected, N]` result with its own routing
//! weights afterwards.
//!
//! ## Offsets are BYTES, like every other grouped kernel here
//!
//! Even though the weights bind as `ushort`. One unit across the whole
//! family beats a per-codec convention nobody can keep straight, so the
//! shader halves the offset itself and the host refuses an odd one — a
//! misaligned `ushort` read would otherwise be silent garbage.
//!
//! ## Two input regimes
//!
//! `XSTRIDE` is `0` when every slot reads the same vector (gate and up,
//! which consume one hidden state) and `K` when each slot reads its own
//! (down, which consumes that expert's own intermediate activation).
//! Explicit rather than inferred: getting it wrong computes a real
//! number from the wrong expert's activation.

/// Row tilings the kernel is emitted for, coarse to fine.
///
/// One simdgroup computes one output row, so a threadgroup of `r` rows
/// needs `r * 32` threads and one launch supplies `ceil(N / r) *
/// n_selected` threadgroups. The cold measurement that earned this
/// sweep: at 1152 threadgroups the `[1024, 2304]` shape sustained 171
/// GB/s and at 2592 the `[2304, 1024]` shape sustained 339 — same bytes,
/// same kernel body, twice the bandwidth from more independent work. So
/// `r` is the knob, and 8 was only ever inherited from `f16_gemv`'s
/// lm_head shape, where `N` is 262144 and threadgroups are never scarce.
pub const ROWS_PER_TG_VARIANTS: [u64; 4] = [8, 4, 2, 1];

/// The default tiling, kept at 8 so the grouped kernel matches
/// [`bf16_gemv`](super::bf16_gemv) and the occupancy comparison that
/// motivated it stays like-for-like.
pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = ROWS_PER_TG * THREADS_PER_SIMDGROUP;

/// Apple-silicon simdgroup width. One row per simdgroup, so this is also
/// the lane stride the reduction body walks `K` with.
pub const THREADS_PER_SIMDGROUP: u64 = 32;

/// Metal kernel function name for a given row tiling.
pub fn kernel_name(rows_per_tg: u64) -> String {
    format!("bf16_grouped_experts_r{rows_per_tg}")
}

/// The MSL for one row tiling. Body identical across variants — only
/// `BF16G_ROWS_PER_TG` and the kernel name change, so every variant must
/// agree bit for bit and any difference between them is scheduling.
pub fn shader_for(rows_per_tg: u64) -> String {
    format!(
        r#"
kernel void {name}(
    device const ushort* W       [[buffer(0)]],  // all expert payloads, bf16 codes
    device const uint*   offsets [[buffer(1)]],  // [n_sel] BYTE offset per slot
    device const float*  X       [[buffer(2)]],  // shared [K], or [n_sel, K]
    device float*        out     [[buffer(3)]],  // [n_sel, N] per-expert outputs
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    // 0 = every slot reads the same X (gate/up). K = each slot reads its
    // own (down). Wrong value = a plausible number from the wrong
    // expert's input, so it is a parameter, not an inferred mode.
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{{
    constexpr uint BF16G_ROWS_PER_TG = {rows};
    // tg_id.y selects the expert slot; tg_id.x the row tile within it.
    const uint slot = tg_id.y;
    const uint row  = tg_id.x * BF16G_ROWS_PER_TG + sg_id;
    if (row >= N) {{ return; }}

    // Offset-table indirection is the whole difference from bf16_gemv.
    // `>> 1` because the table is in bytes and this pointer is ushort.
    device const ushort* w_row = W + (offsets[slot] >> 1u) + (ulong)row * K;
    device const float*  Xs    = X + (ulong)slot * XSTRIDE;

    // ---- body identical to bf16_gemv from here ----
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {{
        a0 = fma(decode_bf16_metal(w_row[k         ]), Xs[k         ], a0);
        a1 = fma(decode_bf16_metal(w_row[k + 32    ]), Xs[k + 32    ], a1);
        a2 = fma(decode_bf16_metal(w_row[k + 64    ]), Xs[k + 64    ], a2);
        a3 = fma(decode_bf16_metal(w_row[k + 96    ]), Xs[k + 96    ], a3);
    }}
    float acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(decode_bf16_metal(w_row[k]), Xs[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[slot * N + row] = acc;
}}
"#,
        name = kernel_name(rows_per_tg),
        rows = rows_per_tg,
    )
}

/// Every variant, concatenated for the shader library.
pub fn shader() -> String {
    ROWS_PER_TG_VARIANTS
        .iter()
        .map(|&r| shader_for(r))
        .collect()
}

/// One `TiledKernel` marker per emitted variant.
///
/// Static markers rather than a runtime lookup so the binding site keeps
/// the crate's no-magic-strings rule — `KernelHandle::from_kernel` reads
/// the name from a constant, and `marker_names_match_the_generator`
/// below pins those constants against [`kernel_name`], which is what the
/// shader generator actually emits. Drift between the two would fail to
/// find the function and return `None` at startup.
macro_rules! grouped_variant {
    ($ident:ident, $rows:literal, $name:literal) => {
        pub struct $ident;
        impl crate::kernels::TiledKernel for $ident {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = $rows;
            const THREADS_PER_TG: u64 = $rows * 32;
        }
    };
}

grouped_variant!(KernelR8, 8, "bf16_grouped_experts_r8");
grouped_variant!(KernelR4, 4, "bf16_grouped_experts_r4");
grouped_variant!(KernelR2, 2, "bf16_grouped_experts_r2");
grouped_variant!(KernelR1, 1, "bf16_grouped_experts_r1");

/// The default tiling's marker, for the binding site that wants only it.
pub type Kernel = KernelR8;

#[cfg(test)]
mod tests {
    use super::*;

    /// The variants must differ ONLY in the tiling constant and the
    /// name. If a body edit reached one and not the others, the "same
    /// arithmetic, different schedule" claim the sweep rests on would be
    /// false — and the sweep would be comparing two kernels.
    #[test]
    fn every_variant_shares_one_body() {
        let normalise = |r: u64| {
            shader_for(r)
                .replace(&kernel_name(r), "K")
                .replace(&format!("= {r};"), "= R;")
        };
        let first = normalise(ROWS_PER_TG_VARIANTS[0]);
        for &r in &ROWS_PER_TG_VARIANTS[1..] {
            assert_eq!(normalise(r), first, "variant r{r} diverged from the body");
        }
    }

    /// The markers and the generator must name the same functions.
    #[test]
    fn marker_names_match_the_generator() {
        use crate::kernels::TiledKernel;
        let declared: Vec<(&str, u64, u64)> = vec![
            (
                KernelR8::KERNEL_NAME,
                KernelR8::ROWS_PER_TG,
                KernelR8::THREADS_PER_TG,
            ),
            (
                KernelR4::KERNEL_NAME,
                KernelR4::ROWS_PER_TG,
                KernelR4::THREADS_PER_TG,
            ),
            (
                KernelR2::KERNEL_NAME,
                KernelR2::ROWS_PER_TG,
                KernelR2::THREADS_PER_TG,
            ),
            (
                KernelR1::KERNEL_NAME,
                KernelR1::ROWS_PER_TG,
                KernelR1::THREADS_PER_TG,
            ),
        ];
        assert_eq!(declared.len(), ROWS_PER_TG_VARIANTS.len());
        for ((name, rows, threads), &r) in declared.iter().zip(&ROWS_PER_TG_VARIANTS) {
            assert_eq!(*rows, r, "marker order must follow ROWS_PER_TG_VARIANTS");
            assert_eq!(*name, kernel_name(r));
            assert_eq!(*threads, r * THREADS_PER_SIMDGROUP);
            assert!(shader().contains(&format!("kernel void {name}(")));
        }
    }

    /// One simdgroup per row is the whole tiling contract; a variant
    /// whose thread count did not follow its row count would drop rows
    /// silently.
    #[test]
    fn threads_follow_rows_one_simdgroup_each() {
        assert_eq!(THREADS_PER_TG, ROWS_PER_TG * THREADS_PER_SIMDGROUP);
        assert!(
            ROWS_PER_TG_VARIANTS.contains(&ROWS_PER_TG),
            "the default tiling must be one of the emitted variants"
        );
        for &r in &ROWS_PER_TG_VARIANTS {
            assert!(
                r.is_power_of_two() && r <= 8,
                "r{r} outside the swept range"
            );
            assert_eq!(kernel_name(r), format!("bf16_grouped_experts_r{r}"));
        }
    }
}
