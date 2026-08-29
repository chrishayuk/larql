//! The router→expert-binding seam, on device.
//!
//! Rung 5d, and the only part of a Kimi decoder layer that is not already
//! a Metal kernel. The residuals and RMS norms are trivial and the crate
//! already has them; the MoE and KDA are done. What was missing is the
//! link that decides WHICH experts run — and it is the link that decides
//! whether a layer is closed at all.
//!
//! ## Why this is the crux
//!
//! The shape to avoid is:
//!
//! ```text
//! GPU router -> read top-8 to the host -> host builds an offset table
//!            -> submit the GPU MoE
//! ```
//!
//! That reintroduces the ~0.23 ms crossing rungs 3 through 5c spent
//! their whole budget removing, once per layer, for the sake of eight
//! integers. The grouped expert kernel already carries expert identity
//! in an **offset table** rather than in a row's position, so the fix is
//! available: if that table lives in a device buffer, a router kernel
//! can write it and the expert kernel can read it later in the same
//! encoder, and the host never learns which experts ran.
//!
//! ## Residency is a separate problem, and this refuses rather than
//! guessing
//!
//! **The router answers WHICH expert, and nothing else.** Where that
//! expert's bytes live is resolved per projection by
//! `kimi_expert_addresses`, because a logical expert may sit at a
//! different physical slot in each of gate/up/down. A single address
//! emitted here would be a shared coordinate across the three, which is
//! exactly the invariant that must not exist.
//!
//! An address, once asked for, can be answered two ways:
//!
//! * **A full execution-shaped bank** holds every expert at its own
//!   index, so the address IS the identity: `offset = expert * stride`.
//!   Nothing is tabulated and no selection can be unaddressable. This is
//!   what a compiled VINDEX writes.
//! * **A packed subset** holds only some experts, so it needs a table,
//!   and a selection outside it has no address. Reading one anyway would
//!   be a plausible wrong answer from another expert's weights, so
//!   `expert_offsets` carries a sentinel and the kernel counts every
//!   refusal into `refusals` for the host to read after the wait — the
//!   same device-counts / host-refuses discipline `route_guard` uses.
//!
//! `identity_stride` selects between them: non-zero means the first,
//! and the table is not consulted at all.
//!
//! The distinction is ADDRESSABILITY, not residency. Whether an
//! addressable expert's pages happen to be resident is a paging concern
//! the model does not express, and conflating the two would make a
//! compiled bank look like a 256-entry residency claim.
//!
//! ## Selection is a threadgroup reduction, and it had to be
//!
//! The first version had one thread run `top_k` passes of argmax over
//! the experts — 2048 comparisons, which looked like nothing beside a
//! 121 MiB expert read. Measured, that single-threaded scan cost **0.98
//! ms**, more than the attention (0.47) and the whole MoE (0.47)
//! together, and turned a layer that should have been faster than the
//! host into 0.86x of it. A GPU core running one thread is not a fast
//! CPU; it is a very slow one.
//!
//! So each pass is a tree reduction over the whole threadgroup, and the
//! tie rule is carried through it explicitly: a candidate wins on a
//! strictly greater score, or on an equal score with a LOWER index.
//! That reproduces `kimi_router::route`'s "descending by selection
//! score, ties by ascending index" exactly rather than approximately —
//! the comparator is associative and commutative under those rules, so
//! the tree and the scan agree.

/// Experts the selection kernel can hold in threadgroup memory. Kimi
/// Linear declares 256; the host refuses more rather than silently
/// truncating a route.
pub const MAX_EXPERTS: usize = 256;
/// Slots the kernel writes: `top_k` routed plus the shared branch.
pub const MAX_SLOTS: usize = 16;
/// One threadgroup, this wide.
pub const SELECT_THREADS_PER_TG: u64 = 256;
/// Marks an expert that is not in the resident bank.
pub const NOT_RESIDENT: u32 = u32::MAX;

pub fn shader() -> String {
    format!(
        r#"
constant uint KIMI_MAX_EXPERTS = {max_experts};
constant uint KIMI_NOT_RESIDENT = {not_resident}u;

// Sigmoid scores, correction-biased selection, deterministic top-k,
// renormalisation, branch scale, and the expert offset table — all in
// one dispatch, so the host never sees a selected id.
//
// MUST be dispatched as ONE threadgroup.
kernel void kimi_router_select(
    device const float* logits      [[buffer(0)]],  // [experts]
    device const float* bias        [[buffer(1)]],  // [experts], SELECTS ONLY
    device float*       scores      [[buffer(2)]],  // [experts] sigmoid(logits), UNBIASED
    device float*       sel_scores  [[buffer(3)]],  // [experts] scores + bias
    device uint*        chosen      [[buffer(4)]],  // [top_k] selected expert ids
    device float*       weights     [[buffer(5)]],  // [top_k + 1] combine weights
    constant uint&      experts     [[buffer(6)]],
    constant uint&      top_k       [[buffer(7)]],
    constant uint&      renormalize [[buffer(8)]],
    constant float&     branch_scale [[buffer(9)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{{
    threadgroup float ts_sel[KIMI_MAX_EXPERTS];
    threadgroup bool  ts_taken[KIMI_MAX_EXPERTS];

    for (uint e = tid; e < experts; e += tcount) {{
        const float s = 1.0f / (1.0f + exp(-logits[e]));
        scores[e] = s;
        const float sel = s + bias[e];
        sel_scores[e] = sel;
        ts_sel[e] = sel;
        ts_taken[e] = false;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // `top_k` passes, each a tree reduction over the whole threadgroup.
    // The comparator prefers a strictly greater score, then a LOWER
    // index — `kimi_router::route`'s rule, and associative under it, so
    // the tree finds what a descending sort would have put first.
    threadgroup float ts_best[KIMI_MAX_EXPERTS];
    threadgroup uint  ts_idx[KIMI_MAX_EXPERTS];
    threadgroup float ts_gathered[{max_slots}];

    for (uint slot = 0u; slot < top_k; ++slot) {{
        ts_best[tid] = (tid < experts && !ts_taken[tid]) ? ts_sel[tid] : -INFINITY;
        ts_idx[tid] = tid;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tcount / 2u; s > 0u; s >>= 1u) {{
            if (tid < s) {{
                const float o = ts_best[tid + s];
                const float m = ts_best[tid];
                if (o > m || (o == m && ts_idx[tid + s] < ts_idx[tid])) {{
                    ts_best[tid] = o;
                    ts_idx[tid] = ts_idx[tid + s];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u) {{
            const uint best = ts_idx[0];
            ts_taken[best] = true;
            chosen[slot] = best;
            // The weight is gathered from the UNBIASED scores. Gathering
            // from the biased ones is the most plausible wrong
            // transcription and changes every routed contribution.
            ts_gathered[slot] = scores[best];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid != 0u) return;
    float sum = 0.0f;
    for (uint slot = 0u; slot < top_k; ++slot) sum += ts_gathered[slot];

    const float denom = (renormalize != 0u && top_k > 1u) ? (sum + 1e-20f) : 1.0f;
    for (uint slot = 0u; slot < top_k; ++slot) {{
        weights[slot] = (ts_gathered[slot] / denom) * branch_scale;
    }}
    // The shared branch: computed like any expert, summed UNSCALED.
    // `routed_scaling_factor` is already folded into the routed weights
    // above and the reference adds the shared output plain. Its ADDRESS
    // is a per-projection fact and is resolved by kimi_expert_addresses.
    weights[top_k] = 1.0f;
}}

// `out[j] = residual[j] + SUM_i weights[i] * experts[i*hidden + j]`,
// with the weights read from a DEVICE buffer.
//
// The crate's `moe_weighted_combine` takes them through `set_bytes` from
// the host, which is correct for a host-side router and exactly the
// dependency this rung removes.
kernel void kimi_moe_combine(
    device const float* experts  [[buffer(0)]],  // [slots, hidden]
    device const float* residual [[buffer(1)]],  // [hidden]
    device const float* weights  [[buffer(2)]],  // [slots]
    device float*       out      [[buffer(3)]],  // [hidden]
    constant uint&      hidden   [[buffer(4)]],
    constant uint&      slots    [[buffer(5)]],
    uint j [[thread_position_in_grid]])
{{
    if (j >= hidden) return;
    float acc = 0.0f;
    for (uint i = 0u; i < slots; ++i) {{
        acc += weights[i] * experts[i * hidden + j];
    }}
    out[j] = residual[j] + acc;
}}

// **Logical expert identity -> byte address, for ONE projection.**
//
// The router says WHICH expert. This says where that expert's bytes are
// FOR THIS PROJECTION, and nothing else does. Dispatched once per
// projection, so gate/up/down each resolve their own coordinate: a
// logical expert may sit at a different physical slot, under a
// different encoding, in each of the three banks, and no shared
// coordinate survives anywhere in the model.
//
// `stride != 0` addresses by identity (`slot == logical id`), the shape
// a compiled execution-ordered bank has. `stride == 0` consults the
// table, which is what an arbitrarily-ordered source bank needs. A
// selection the table has no address for is counted here — addressing
// is where that fact lives, not routing.
kernel void kimi_expert_addresses(
    device const uint*  chosen        [[buffer(0)]],  // [top_k] logical ids
    device const uint*  table         [[buffer(1)]],  // [experts]; unread when stride != 0
    device uint*        offsets       [[buffer(2)]],  // [top_k + 1] byte offsets
    device atomic_uint* refusals      [[buffer(3)]],
    constant uint&      top_k         [[buffer(4)]],
    constant uint&      stride        [[buffer(5)]],
    constant uint&      shared_offset [[buffer(6)]],
    constant uint&      experts       [[buffer(7)]],
    uint slot [[thread_position_in_grid]])
{{
    if (slot > top_k) return;
    if (slot == top_k) {{
        // The shared branch is not routed and has no logical id; its
        // address is a property of this projection's bank.
        offsets[slot] = shared_offset;
        return;
    }}
    const uint id = chosen[slot];
    if (id >= experts) {{
        atomic_fetch_add_explicit(refusals, 1u, memory_order_relaxed);
        offsets[slot] = 0u;
        return;
    }}
    const uint off = stride != 0u ? id * stride : table[id];
    if (off == KIMI_NOT_RESIDENT) {{
        // No address for this expert in THIS projection. Reading anyway
        // would be a plausible wrong answer from another expert's
        // weights; bind slot 0 to stay in bounds and let the host refuse
        // after the wait.
        atomic_fetch_add_explicit(refusals, 1u, memory_order_relaxed);
        offsets[slot] = 0u;
    }} else {{
        offsets[slot] = off;
    }}
}}
"#,
        max_experts = MAX_EXPERTS,
        max_slots = MAX_SLOTS,
        not_resident = NOT_RESIDENT,
    )
}

pub struct RouterSelectKernel;
impl crate::kernels::ShaderKernel for RouterSelectKernel {
    const KERNEL_NAME: &'static str = "kimi_router_select";
}

pub struct ExpertAddressesKernel;
impl crate::kernels::ShaderKernel for ExpertAddressesKernel {
    const KERNEL_NAME: &'static str = "kimi_expert_addresses";
}

pub struct MoeCombineKernel;
impl crate::kernels::ShaderKernel for MoeCombineKernel {
    const KERNEL_NAME: &'static str = "kimi_moe_combine";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ShaderKernel;

    /// Both kernels are emitted, and the threadgroup arrays are sized
    /// from the same constants the host validates against — a router
    /// dispatched with more experts than `ts_sel` holds would write out
    /// of bounds.
    #[test]
    fn the_emitted_source_matches_the_declared_limits() {
        let src = shader();
        for name in [
            RouterSelectKernel::KERNEL_NAME,
            MoeCombineKernel::KERNEL_NAME,
        ] {
            assert!(
                src.contains(&format!("kernel void {name}(")),
                "{name} missing"
            );
        }
        assert!(src.contains(&format!("constant uint KIMI_MAX_EXPERTS = {MAX_EXPERTS};")));
        assert!(src.contains(&format!("threadgroup float ts_gathered[{MAX_SLOTS}];")));
        assert!(src.contains(&format!(
            "constant uint KIMI_NOT_RESIDENT = {NOT_RESIDENT}u;"
        )));
        assert!(
            src.contains("threadgroup float ts_sel[KIMI_MAX_EXPERTS];"),
            "the selection buffer must be sized by the declared maximum"
        );
    }

    /// The weight must be gathered from the UNBIASED scores. Gathering
    /// from the biased array preserves the selection and silently
    /// changes every routed contribution, which is the failure this
    /// whole seam is designed to make impossible — so it is pinned in
    /// the source, not only in a runtime gate.
    #[test]
    fn the_weight_is_gathered_from_the_unbiased_scores() {
        let src = shader();
        assert!(src.contains("ts_gathered[slot] = scores[best];"));
        assert!(
            !src.contains("ts_gathered[slot] = sel_scores[best];"),
            "weights must never be gathered from the biased selection scores"
        );
    }

    /// Ties resolve to the lowest index, matching `kimi_router::route`.
    /// The comparator is the whole reason the tree agrees with a sort.
    #[test]
    fn selection_breaks_ties_towards_the_lowest_index() {
        assert!(shader().contains("if (o > m || (o == m && ts_idx[tid + s] < ts_idx[tid]))"));
    }
}
