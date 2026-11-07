//! VERIFY-N routing for a routed layer: which layers can run as a
//! `[rows, hidden]` block, and the block's route in one pass.

use super::options::DedupeArm;
use super::transform::{router_input_transform, RouterInputTransform};
use crate::moe_dispatch::MoeScratch;
use crate::shaders::moe_router::SLOTS_BY_EXPERT_THREADS;
use crate::shaders::moe_router_select::{MAX_EXPERTS, MAX_TOP_K, TG_THREADS};
use crate::shaders::mxfp4_grouped_experts::MULTI_MAX_SLOTS;
use crate::MetalBackend;
use larql_compute::{MoeExpertScalePolicy, MoeLayerWeights, MoeTopKWeightPolicy};
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use std::ffi::c_void;

/// A block's route, GPU-resident: the normalised router/expert input
/// `x` (`[rows, hidden]`), the router `logits` (`[rows, E]`), and the
/// selected expert `ids` / `weights` (`[rows, top_k]`).
pub(crate) struct RouteRows {
    pub(crate) x: Buffer,
    // Read by the routing parity gate; the expert pass consumes ids/weights.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) logits: Buffer,
    pub(crate) ids: Buffer,
    pub(crate) weights: Buffer,
}

impl MetalBackend {
    /// Whether a layer's routed FFN can run as a VERIFY-N block through
    /// [`Self::encode_moe_layer_gpu_route_at`]: its router input must be the
    /// pre-experts norm (see there). `Err` names the reason.
    pub fn routed_rows_supported(&self, moe: &MoeLayerWeights<'_>) -> Result<(), String> {
        match router_input_transform(moe) {
            Some(RouterInputTransform::PreExpertsRmsNorm) => Ok(()),
            other => Err(format!(
                "multi-position routed FFN needs a pre-experts-norm router input \
                 (this layer's is {other:?}: the residual itself is the expert input, \
                 bound at offset 0)"
            )),
        }
    }

    /// Route every row of a `[rows, hidden]` block in three dispatches —
    /// pre-experts norm, router logits, top-k select — where the per-row
    /// path spends three per row. Each row's `x`, logits, ids and weights
    /// are bit-identical to that row routed alone: the rows kernels run
    /// the single kernels' bodies at row offsets, at the same widths.
    pub(crate) fn encode_moe_route_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &MoeLayerWeights<'_>,
        h_rows: &Buffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<RouteRows, String> {
        self.routed_rows_supported(moe)?;
        let (num_experts, top_k) = (moe.num_experts, moe.top_k);
        if num_experts == 0
            || num_experts > MAX_EXPERTS
            || top_k == 0
            || top_k > MAX_TOP_K
            || moe.router_proj.len() != num_experts * hidden
        {
            return Err(format!(
                "routed block: router shape outside the select kernel \
                 (experts {num_experts}, top-k {top_k}, hidden {hidden})"
            ));
        }
        let f = std::mem::size_of::<f32>() as u64;

        // Pre-experts norm, offset 0.0 as the single-position encode.
        let x = self.bufs.output(rows as u64 * hidden as u64 * f);
        let norm_w = self.bufs.get_f32(moe.pre_experts_norm);
        self.encode_rms_norm_rows(enc, h_rows, 0, &norm_w, &x, 0, hidden, rows, eps, 0.0);

        // Router logits, grid (E, rows).
        let logits = self.bufs.output(rows as u64 * num_experts as u64 * f);
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let bias = (!moe.router_bias.is_empty()).then(|| self.bufs.get_f32(moe.router_bias));
        let (e_u, h_u) = (num_experts as u32, hidden as u32);
        let has_bias = u32::from(bias.is_some());
        enc.set_compute_pipeline_state(&self.ffn.moe_router_rows_pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x), 0);
        // Placeholder on the has_bias = 0 arm, never read.
        enc.set_buffer(2, Some(bias.as_ref().unwrap_or(&x)), 0);
        enc.set_buffer(3, Some(&logits), 0);
        enc.set_bytes(4, 4, &e_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &h_u as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &has_bias as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_experts as u64, rows as u64, 1),
            MTLSize::new(crate::shaders::moe_router::THREADS_PER_TG, 1, 1),
        );

        // Top-k select, one threadgroup per row.
        let ids = self.bufs.output(rows as u64 * top_k as u64 * 4);
        let weights = self.bufs.output(rows as u64 * top_k as u64 * f);
        let renorm = u32::from(
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax,
        );
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then(|| self.bufs.get_f32(moe.router_per_expert_scale));
        let has_scale = u32::from(pe_scale.is_some());
        let k_u = top_k as u32;
        enc.set_compute_pipeline_state(&self.ffn.moe_router_select_rows_pipeline);
        enc.set_buffer(0, Some(&logits), 0);
        enc.set_buffer(1, Some(pe_scale.as_ref().unwrap_or(&logits)), 0);
        enc.set_buffer(2, Some(&ids), 0);
        enc.set_buffer(3, Some(&weights), 0);
        enc.set_bytes(4, 4, &e_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_u as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &renorm as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &has_scale as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(TG_THREADS, 1, 1),
        );
        Ok(RouteRows {
            x,
            logits,
            ids,
            weights,
        })
    }
}

/// Device scratch for a routed layer run as a VERIFY-N block (rung 5a):
/// one expert slot per (row, selected expert) — a [`MoeScratch`] of
/// `rows × top_k` slots — plus the block's input laid out one row per
/// slot at the layer's stored (padded) row width, with the padding kept
/// zero. Shared by every routed layer of the stack under the serial
/// encoder, as the per-layer scratch is shared by the rows of rung 2.
pub struct MoeBlockScratch {
    rows: usize,
    pub(crate) slots: MoeScratch,
    pub(crate) x_slots: Buffer,
}

impl MoeBlockScratch {
    /// Block scratch for up to `rows` positions of layers shaped like
    /// `layer` (its top-k, widths and expert format).
    pub fn for_layer(backend: &MetalBackend, layer: &MoeScratch, rows: usize) -> Self {
        let n_slots = rows * layer.top_k;
        let slots = MoeScratch::new(
            &backend.bufs,
            n_slots,
            layer.hidden,
            layer.inter,
            layer.format,
            layer.weight_cols,
        );
        let len = n_slots * layer.weight_cols;
        let x_slots = backend
            .bufs
            .output((len * std::mem::size_of::<f32>()) as u64);
        // SAFETY: a shared-storage buffer of exactly `len` f32s, not yet
        // bound to any command buffer. Pooled buffers arrive dirty; the
        // padding columns must be zero because no kernel ever writes them.
        unsafe { std::ptr::write_bytes(x_slots.contents() as *mut f32, 0, len) };
        Self {
            rows,
            slots,
            x_slots,
        }
    }

    /// Whether this scratch serves `rows` positions of a layer shaped like
    /// `layer`.
    pub fn fits(&self, layer: &MoeScratch, rows: usize) -> bool {
        rows <= self.rows
            && self.slots.top_k == self.rows * layer.top_k
            && self.slots.hidden == layer.hidden
            && self.slots.inter == layer.inter
            && self.slots.weight_cols == layer.weight_cols
            && self.slots.format == layer.format
    }
}

impl MetalBackend {
    /// **Rung 5a.** A routed layer over a `[rows, hidden]` block in one
    /// pass: route every row ([`Self::encode_moe_route_rows`]), lay the
    /// input out one row per slot, and run the grouped expert pass over
    /// all `rows × top_k` slots at once — each slot's arithmetic exactly
    /// the single position's — then combine each row into its own residual.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_layer_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &MoeLayerWeights<'_>,
        layer: &MoeScratch,
        block: &MoeBlockScratch,
        table: &crate::moe_descriptor::MoeExpertDescriptorTable,
        (h_rows, new_h): (&Buffer, &Buffer),
        rows: usize,
        eps: f32,
    ) -> Result<(), String> {
        if !block.fits(layer, rows) {
            return Err(format!(
                "routed block scratch does not fit {rows} rows of this layer"
            ));
        }
        let route = self.encode_moe_route_rows(enc, moe, h_rows, rows, layer.hidden, eps)?;
        let n_slots = rows * layer.top_k;
        let (h_u, k_u, c_u) = (
            layer.hidden as u32,
            layer.top_k as u32,
            layer.weight_cols as u32,
        );
        enc.set_compute_pipeline_state(&self.ffn.moe_rows_to_slots_pipeline);
        enc.set_buffer(0, Some(&route.x), 0);
        enc.set_buffer(1, Some(&block.x_slots), 0);
        enc.set_bytes(2, 4, &h_u as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &k_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &c_u as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(layer.hidden as u64, n_slots as u64, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(layer.hidden as u64),
                1,
                1,
            ),
        );
        self.encode_experts_and_combine_descriptor_x_buf_at(
            enc,
            &block.x_slots,
            moe,
            &block.slots,
            table,
            &route.ids,
            &route.weights,
            (h_rows, 0),
            (new_h, 0),
            super::encode::SlotLayout {
                n_slots,
                x_stride: c_u,
                top_k: layer.top_k,
                dedupe: (rows > 1 && rows <= MULTI_MAX_SLOTS)
                    .then(super::options::routed_dedupe_arm)
                    .flatten(),
            },
        );
        Ok(())
    }
}

/// A block's slots grouped by distinct expert, on the device (rung 5b):
/// `counts[u]` slots of distinct expert `u`, listed at `lists[u * maxm ..]`.
pub(crate) struct SlotGroups {
    arm: DedupeArm,
    counts: Buffer,
    lists: Buffer,
    order: Buffer,
    maxm: u32,
}

impl SlotGroups {
    /// The grouped expert kernel this arm runs in place of x2.
    pub(crate) fn pipeline<'a>(
        &self,
        backend: &'a MetalBackend,
    ) -> &'a crate::kernels::KernelHandle {
        match self.arm {
            DedupeArm::Sorted => &backend.quant.mxfp4_grouped_x2_sorted_pipeline,
            DedupeArm::Multi => &backend.quant.mxfp4_grouped_x2_multi_pipeline,
        }
    }

    /// Bind the arm's extra buffers (11.. after x2's table).
    pub(crate) fn bind(&self, enc: &ComputeCommandEncoderRef) {
        match self.arm {
            DedupeArm::Sorted => enc.set_buffer(11, Some(&self.order), 0),
            DedupeArm::Multi => {
                enc.set_buffer(11, Some(&self.counts), 0);
                enc.set_buffer(12, Some(&self.lists), 0);
                enc.set_bytes(13, 4, &self.maxm as *const u32 as *const c_void);
            }
        }
    }

    /// The arm's grid for `row_tiles` tiles over `n_slots` slots: the
    /// sorted arm puts slots on x so expert-sharing threadgroups neighbour.
    pub(crate) fn grid(&self, row_tiles: u64, n_slots: usize) -> MTLSize {
        match self.arm {
            DedupeArm::Sorted => MTLSize::new(n_slots as u64, row_tiles, 1),
            DedupeArm::Multi => MTLSize::new(row_tiles, n_slots as u64, 1),
        }
    }
}

impl MetalBackend {
    /// Group `n_slots` routed slots (`ids`) by expert: one dispatch, one
    /// threadgroup. `maxm` is the most slots an expert can carry — the
    /// block's rows, since an expert appears at most once per row.
    pub(crate) fn encode_slots_by_expert(
        &self,
        enc: &ComputeCommandEncoderRef,
        ids: &Buffer,
        n_slots: usize,
        num_experts: usize,
        maxm: usize,
        arm: DedupeArm,
    ) -> SlotGroups {
        assert!(
            n_slots as u64 <= SLOTS_BY_EXPERT_THREADS
                && num_experts as u64 <= SLOTS_BY_EXPERT_THREADS
                && maxm <= MULTI_MAX_SLOTS,
            "slot grouping bounds: {n_slots} slots, {num_experts} experts, {maxm} per expert"
        );
        let counts = self.bufs.output((n_slots * 4) as u64);
        let lists = self.bufs.output((n_slots * maxm * 4) as u64);
        let order = self.bufs.output((n_slots * 4) as u64);
        let (s_u, e_u, m_u) = (n_slots as u32, num_experts as u32, maxm as u32);
        enc.set_compute_pipeline_state(&self.ffn.moe_slots_by_expert_pipeline);
        enc.set_buffer(0, Some(ids), 0);
        enc.set_buffer(1, Some(&counts), 0);
        enc.set_buffer(2, Some(&lists), 0);
        enc.set_bytes(3, 4, &s_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &e_u as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &m_u as *const u32 as *const c_void);
        enc.set_buffer(6, Some(&order), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(SLOTS_BY_EXPERT_THREADS, 1, 1),
        );
        SlotGroups {
            arm,
            counts,
            lists,
            order,
            maxm: m_u,
        }
    }
}
