//! Format-agnostic GPU-route layer encode + the per-format kernel
//! bindings (selected by `scratch.format` + the kernel registry's
//! `ExpertScaleBinding` — architecture facts, never model names).

use super::transform::{router_input_transform, RouterInputTransform};
use crate::moe_descriptor::MoeExpertDescriptorTable;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;
use metal::{Buffer, MTLSize};
use std::ffi::c_void;

impl MetalBackend {
    /// Encode the full expert block + combine with descriptor-driven
    /// bindings. `selected_ids` / `selected_weights` are rung B's
    /// GPU-resident route result; the CPU never reads them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_experts_and_combine_descriptor(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        selected_ids: &Buffer,
        selected_weights: &Buffer,
        h_post_attn: &Buffer,
        new_h: &Buffer,
    ) {
        // Route-INDEPENDENT x staging (same bytes whichever experts win).
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, scratch.hidden);
        }
        self.encode_experts_and_combine_descriptor_x_buf(
            enc,
            &scratch.x_buf.clone(),
            moe,
            scratch,
            table,
            selected_ids,
            selected_weights,
            h_post_attn,
            new_h,
        );
    }

    /// Core encode with the expert input already GPU-resident — the form
    /// a pre-encoded token chain needs, where layer i+1's x IS layer i's
    /// `new_h` buffer and no host staging may sit between them. When
    /// `x_buf` is not the scratch's padded staging buffer, the stored row
    /// width must equal `hidden` (no zero tail to rely on).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_experts_and_combine_descriptor_x_buf(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        selected_ids: &Buffer,
        selected_weights: &Buffer,
        h_post_attn: &Buffer,
        new_h: &Buffer,
    ) {
        assert!(
            x_buf.gpu_address() == scratch.x_buf.gpu_address()
                || scratch.weight_cols == scratch.hidden,
            "chained x binding needs weight_cols == hidden: a padded row \
             width relies on the staging buffer's zero tail"
        );
        assert!(
            matches!(
                scratch.format,
                larql_compute::QuantFormat::Q6_K | larql_compute::QuantFormat::MXFP4
            ),
            "descriptor arm serves Q6_K and native MXFP4; other formats stay \
             on the legacy path by explicit caller choice"
        );
        if scratch.format == larql_compute::QuantFormat::Q6_K {
            // Q6_K selects its up half by byte offset (gate1 = gate0 +
            // gate_half_bytes), which can only describe ContiguousHalves.
            // MXFP4 selects halves by ROW WALK and serves either layout.
            assert_eq!(
                moe.fused_row_layout,
                larql_compute::MoeFusedRowLayout::ContiguousHalves,
                "the Q6_K descriptor arm cannot serve a {:?} bank",
                moe.fused_row_layout,
            );
        }
        debug_assert!(matches!(
            moe.routing_policy.post_expert_norm,
            larql_compute::MoePostExpertNormPolicy::None
        ));
        let hidden = scratch.hidden;
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        let n_slots = scratch.top_k;
        let gate_half_bytes = inter * scratch.row_bytes;
        assert_eq!(
            table.gate_up_expert_bytes,
            2 * gate_half_bytes,
            "descriptor table's expert size disagrees with the scratch dims"
        );

        // The single runtime indirection: route → stored-expert bindings.
        let bindings = self.encode_descriptor_gather(
            enc,
            table,
            selected_ids,
            n_slots,
            gate_half_bytes as u32,
        );

        let ablate = moe_ablation();
        // E3: bias presence is a layer fact, stated by the table.
        let stage_gate_up_bias = table.gate_up_bias_bank.is_some() && !ablate.bias;
        if let Some(bank) = &table.gate_up_bias_bank {
            self.encode_bias_stage(
                enc,
                bank,
                &bindings.slot_descs,
                (&scratch.gate_bias_buf, &scratch.up_bias_buf),
                inter,
                n_slots,
            );
        }

        // Gate + up halves: the D-proven binding — same grouped kernels as
        // the legacy path, offsets/scale-offsets from the gathered buffers
        // instead of set_bytes.
        let n_rows = inter as u32;
        let k_cols = scratch.weight_cols as u32;
        let xstride_shared: u32 = 0;
        match scratch.format {
            larql_compute::QuantFormat::MXFP4 => {
                use larql_models::quant::mxfp4::FusedHalf;
                let (kh, binding) = self.mxfp4_grouped_for_table(table);
                assert_eq!(
                    binding,
                    crate::kernels::quant::ExpertScaleBinding::SplitE8M0,
                    "descriptor MXFP4 arm needs the split-scale kernel"
                );
                let scale_base = table
                    .gate_up_scale_base
                    .as_ref()
                    .expect("gpu_route_supported checked the scale streams");
                let fused_gu = std::ptr::eq(
                    kh as *const _,
                    &self.quant.mxfp4_grouped_x2_pipeline as *const _,
                ) && expert_gu_fusion_enabled();
                if fused_gu {
                    // A-12 expert pass: both halves in ONE dispatch — one
                    // GEMV α per layer instead of two, and twice the
                    // threadgroups the x2 arm alone launches. Bit-identical
                    // per row to the two-dispatch form.
                    let khgu = &self.quant.mxfp4_grouped_x2_gu_pipeline;
                    let (g_base, g_stride) = moe.fused_row_layout.row_walk(FusedHalf::Gate, inter);
                    let (u_base, u_stride) = moe.fused_row_layout.row_walk(FusedHalf::Up, inter);
                    let (g_base, g_stride) = (g_base as u32, g_stride as u32);
                    let (u_base, u_stride) = (u_base as u32, u_stride as u32);
                    let row_tiles = (2 * inter as u64).div_ceil(khgu.rows_per_tg);
                    enc.set_compute_pipeline_state(&khgu.state);
                    enc.set_buffer(0, Some(&table.gate_up_base), 0);
                    enc.set_buffer(1, Some(&bindings.gate0_offs), 0);
                    enc.set_buffer(2, Some(scale_base), 0);
                    enc.set_buffer(3, Some(&bindings.gu_scale_offs), 0);
                    enc.set_buffer(4, Some(x_buf), 0);
                    enc.set_buffer(5, Some(&scratch.g_out), 0);
                    enc.set_bytes(6, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &k_cols as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &xstride_shared as *const u32 as *const c_void);
                    enc.set_bytes(9, 4, &g_base as *const u32 as *const c_void);
                    enc.set_bytes(10, 4, &g_stride as *const u32 as *const c_void);
                    enc.set_buffer(11, Some(&scratch.u_out), 0);
                    enc.set_bytes(12, 4, &u_base as *const u32 as *const c_void);
                    enc.set_bytes(13, 4, &u_stride as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(row_tiles, n_slots as u64, 1),
                        MTLSize::new(khgu.threads_per_tg, 1, 1),
                    );
                } else {
                    let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);
                    for half in [FusedHalf::Gate, FusedHalf::Up] {
                        // Halves are selected by WHICH ROWS the kernel
                        // walks — one payload table and one scale table
                        // serve both.
                        let (row_base, row_stride) = moe.fused_row_layout.row_walk(half, inter);
                        let (row_base, row_stride) = (row_base as u32, row_stride as u32);
                        let out_buf = match half {
                            FusedHalf::Gate => &scratch.g_out,
                            FusedHalf::Up => &scratch.u_out,
                        };
                        enc.set_compute_pipeline_state(&kh.state);
                        enc.set_buffer(0, Some(&table.gate_up_base), 0);
                        enc.set_buffer(1, Some(&bindings.gate0_offs), 0);
                        enc.set_buffer(2, Some(scale_base), 0);
                        enc.set_buffer(3, Some(&bindings.gu_scale_offs), 0);
                        enc.set_buffer(4, Some(x_buf), 0);
                        enc.set_buffer(5, Some(out_buf), 0);
                        enc.set_bytes(6, 4, &n_rows as *const u32 as *const c_void);
                        enc.set_bytes(7, 4, &k_cols as *const u32 as *const c_void);
                        enc.set_bytes(8, 4, &xstride_shared as *const u32 as *const c_void);
                        enc.set_bytes(9, 4, &row_base as *const u32 as *const c_void);
                        enc.set_bytes(10, 4, &row_stride as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(row_tiles, n_slots as u64, 1),
                            MTLSize::new(kh.threads_per_tg, 1, 1),
                        );
                    }
                }
            }
            _ => {
                let kh = &self.quant.q6k_grouped_experts_pipeline;
                let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);
                for (offs, out_buf) in [
                    (&bindings.gate0_offs, &scratch.g_out),
                    (&bindings.gate1_offs, &scratch.u_out),
                ] {
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(&table.gate_up_base), 0);
                    enc.set_buffer(1, Some(offs), 0);
                    enc.set_buffer(2, Some(x_buf), 0);
                    enc.set_buffer(3, Some(out_buf), 0);
                    enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_cols as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &xstride_shared as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(row_tiles, n_slots as u64, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
        }

        // Activation — slot-shaped, route-independent (identical to the
        // legacy path; only the bias-presence authority changed).
        let inter_u32 = inter as u32;
        for e in 0..if ablate.act { 0 } else { n_slots } {
            let g_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_gate_up_bias);
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                    enc.set_buffer(4, Some(&scratch.gate_bias_buf), g_offset);
                    enc.set_buffer(5, Some(&scratch.up_bias_buf), g_offset);
                    enc.set_bytes(6, 4, &has_bias as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &limit as *const f32 as *const c_void);
                    enc.set_bytes(8, 4, &alpha as *const f32 as *const c_void);
                }
                larql_compute::MoeGateRule::SituGlu { beta, linear_beta } => {
                    crate::kernels::ffn::bind_situ_glu(
                        enc,
                        &self.ffn.situ_glu_pipeline,
                        (&scratch.g_out, g_offset),
                        (&scratch.u_out, g_offset),
                        (&scratch.act_buf, a_offset),
                        inter_u32,
                        beta,
                        linear_beta,
                        stage_gate_up_bias,
                    );
                }
                larql_compute::MoeGateRule::Gated(activation) => {
                    let pipeline = if activation.gate_up_is_gelu_tanh() {
                        &self.ffn.geglu_gelu_tanh_pipeline
                    } else {
                        &self.ffn.geglu_pipeline
                    };
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                }
            }
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // Down projection: grouped, each slot reading its own activation.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let xstride_own: u32 = inter_padded as u32;
        // A-12: for a top-4 MXFP4 route the down projection and the
        // weighted combine fuse into one dispatch (bit-identical: same
        // per-(row,slot) walk, same combine order). The staged down bias
        // moves BEFORE it; the separate combine kernel is skipped.
        let fused_down_combine = scratch.format == larql_compute::QuantFormat::MXFP4
            && n_slots == 4
            && table.payload_offsets_vec16
            && expert_down_combine_enabled()
            && !ablate.combine;
        if fused_down_combine {
            let has_down_bias = table.down_bias_bank.is_some() && !ablate.bias;
            if let Some(bank) = &table.down_bias_bank {
                if !ablate.bias {
                    let hidden_u32 = hidden as u32;
                    let n = n_slots as u32;
                    enc.set_compute_pipeline_state(&self.moe_down_bias_stage_pipeline);
                    enc.set_buffer(0, Some(bank), 0);
                    enc.set_buffer(1, Some(&bindings.slot_descs), 0);
                    enc.set_buffer(2, Some(&scratch.down_bias_staged), 0);
                    enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                    enc.dispatch_threads(
                        MTLSize::new(hidden as u64, n_slots as u64, 1),
                        MTLSize::new(64.min(hidden as u64).max(1), 1, 1),
                    );
                }
            }
            let scale_base = table
                .down_scale_base
                .as_ref()
                .expect("gpu_route_supported checked the scale streams");
            let khdc = &self.quant.mxfp4_down_combine4_pipeline;
            let has_bias_u: u32 = u32::from(has_down_bias);
            enc.set_compute_pipeline_state(&khdc.state);
            enc.set_buffer(0, Some(&table.down_base), 0);
            enc.set_buffer(1, Some(&bindings.down_offs), 0);
            enc.set_buffer(2, Some(scale_base), 0);
            enc.set_buffer(3, Some(&bindings.dn_scale_offs), 0);
            enc.set_buffer(4, Some(&scratch.act_buf), 0);
            enc.set_buffer(5, Some(new_h), 0);
            enc.set_bytes(6, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &k_in as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &xstride_own as *const u32 as *const c_void);
            enc.set_buffer(9, Some(h_post_attn), 0);
            enc.set_buffer(10, Some(selected_weights), 0);
            enc.set_buffer(11, Some(&scratch.down_bias_staged), 0);
            enc.set_bytes(12, 4, &has_bias_u as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(khdc.rows_per_tg), 1, 1),
                MTLSize::new(khdc.threads_per_tg, 1, 1),
            );
            return;
        }
        match scratch.format {
            larql_compute::QuantFormat::MXFP4 => {
                use crate::shaders::mxfp4_grouped_experts::{
                    ROW_BASE_IDENTITY, ROW_STRIDE_IDENTITY,
                };
                let (kh, _) = self.mxfp4_grouped_for_table(table);
                let scale_base = table
                    .down_scale_base
                    .as_ref()
                    .expect("gpu_route_supported checked the scale streams");
                let row_tiles_down = (hidden as u64).div_ceil(kh.rows_per_tg);
                enc.set_compute_pipeline_state(&kh.state);
                enc.set_buffer(0, Some(&table.down_base), 0);
                enc.set_buffer(1, Some(&bindings.down_offs), 0);
                enc.set_buffer(2, Some(scale_base), 0);
                enc.set_buffer(3, Some(&bindings.dn_scale_offs), 0);
                enc.set_buffer(4, Some(&scratch.act_buf), 0);
                enc.set_buffer(5, Some(&scratch.expert_outs), 0);
                enc.set_bytes(6, 4, &n_out as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &k_in as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &xstride_own as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &ROW_BASE_IDENTITY as *const u32 as *const c_void);
                enc.set_bytes(10, 4, &ROW_STRIDE_IDENTITY as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(row_tiles_down, n_slots as u64, 1),
                    MTLSize::new(kh.threads_per_tg, 1, 1),
                );
            }
            _ => {
                let kh = &self.quant.q6k_grouped_experts_pipeline;
                let row_tiles_down = (hidden as u64).div_ceil(kh.rows_per_tg);
                enc.set_compute_pipeline_state(&kh.state);
                enc.set_buffer(0, Some(&table.down_base), 0);
                enc.set_buffer(1, Some(&bindings.down_offs), 0);
                enc.set_buffer(2, Some(&scratch.act_buf), 0);
                enc.set_buffer(3, Some(&scratch.expert_outs), 0);
                enc.set_bytes(4, 4, &n_out as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &k_in as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &xstride_own as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(row_tiles_down, n_slots as u64, 1),
                    MTLSize::new(kh.threads_per_tg, 1, 1),
                );
            }
        }

        // Down biases: descriptor-driven staging into the same scratch the
        // combine kernel reads (E1 — the last route-dependent CPU memcpy).
        let has_down_bias = table.down_bias_bank.is_some() && !ablate.bias;
        if let Some(bank) = &table.down_bias_bank {
            let hidden_u32 = hidden as u32;
            let n = n_slots as u32;
            enc.set_compute_pipeline_state(&self.moe_down_bias_stage_pipeline);
            enc.set_buffer(0, Some(bank), 0);
            enc.set_buffer(1, Some(&bindings.slot_descs), 0);
            enc.set_buffer(2, Some(&scratch.down_bias_staged), 0);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, n_slots as u64, 1),
                MTLSize::new(64.min(hidden as u64).max(1), 1, 1),
            );
        }

        if ablate.combine {
            return;
        }
        // Combine — same kernel, routing weights from rung B's GPU buffer
        // (E2: the set_bytes → set_buffer flip, kernel signature unchanged).
        let hidden_u = hidden as u32;
        let k_u = n_slots as u32;
        let has_bias_u: u32 = u32::from(has_down_bias);
        enc.set_compute_pipeline_state(&self.ffn.moe_weighted_combine_pipeline);
        enc.set_buffer(0, Some(&scratch.expert_outs), 0);
        enc.set_buffer(1, Some(h_post_attn), 0);
        enc.set_buffer(2, Some(new_h), 0);
        enc.set_bytes(3, 4, &hidden_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const c_void);
        enc.set_buffer(5, Some(selected_weights), 0);
        enc.set_buffer(6, Some(&scratch.down_bias_staged), 0);
        enc.set_bytes(7, 4, &has_bias_u as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }
}

impl MetalBackend {
    /// The grouped MXFP4 kernel to encode this table with: the
    /// options-selected arm, demoted to the scalar split arm when the
    /// vectorised arm's 16-byte payload-alignment precondition does not
    /// hold for this bank. The demotion was already announced once at
    /// table build; here it must only be correct.
    fn mxfp4_grouped_for_table(
        &self,
        table: &MoeExpertDescriptorTable,
    ) -> (
        &crate::kernels::KernelHandle,
        crate::kernels::quant::ExpertScaleBinding,
    ) {
        use crate::shaders::mxfp4_grouped_experts::Mxfp4Arm;
        let (kh, binding) = self
            .quant
            .grouped_experts_for(larql_compute::QuantFormat::MXFP4);
        if self.quant.mxfp4_grouped_arm == Mxfp4Arm::SplitLut16Vec && !table.payload_offsets_vec16 {
            return (&self.quant.mxfp4g_split_lut16_pipeline, binding);
        }
        // A-12 expert pass: the x2 arm (two rows per simdgroup sharing X
        // loads) is bit-identical per row and needs the same 16-byte
        // alignment as the vec arm. Decomposition bench: 262 → 313 GB/s
        // at the gpt-oss expert shape (`examples/moe_expert_alpha_b.rs`).
        // `LARQL_MXFP4_EXPERT_X2=0` restores the single-row arm (control).
        if self.quant.mxfp4_grouped_arm == Mxfp4Arm::SplitLut16Vec && expert_x2_enabled() {
            return (&self.quant.mxfp4_grouped_x2_pipeline, binding);
        }
        (kh, binding)
    }

    /// Everything the GPU route needs to hold, checked BEFORE any
    /// command-buffer state is touched — a `false` here means the CPU
    /// arm proceeds with nothing to roll back.
    pub(crate) fn gpu_route_supported(
        &self,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
    ) -> bool {
        let format_ok = match scratch.format {
            larql_compute::QuantFormat::Q6_K => {
                moe.fused_row_layout == larql_compute::MoeFusedRowLayout::ContiguousHalves
            }
            // MXFP4 halves select by row walk (either layout), but the bank
            // must be split-scale and the kernel registry must have the
            // split-binding arm selected.
            larql_compute::QuantFormat::MXFP4 => {
                moe.expert_scales.is_paired()
                    && self.quant.grouped_experts_for(scratch.format).1
                        == crate::kernels::quant::ExpertScaleBinding::SplitE8M0
            }
            _ => false,
        };
        // SiTU-GLU has no bias form in the reference (K3's experts are
        // bias-free `nn.Linear`s) and `situ_glu` has no bias slots, so a
        // layer staging per-expert gate/up biases under it is refused
        // HERE — at admission, before an encoder exists — rather than at
        // the dispatch. `bind_situ_glu` still asserts, as the backstop for
        // any path that does not come through this gate, but a refusal
        // raised mid-encode leaves the encoder unended and Metal aborts
        // the process instead of reporting it.
        let situ_bias_ok = !matches!(moe.gate_rule, larql_compute::MoeGateRule::SituGlu { .. })
            || moe.experts_gate_up_bias.is_empty();
        situ_bias_ok
            && router_input_transform(moe).is_some()
            && format_ok
            && matches!(
                moe.routing_policy.post_expert_norm,
                larql_compute::MoePostExpertNormPolicy::None
            )
            && moe.num_experts <= crate::shaders::moe_router_select::MAX_EXPERTS
            && moe.top_k >= 1
            && moe.top_k <= crate::shaders::moe_router_select::MAX_TOP_K
            && moe.router_proj.len() == moe.num_experts * scratch.hidden
            // Identity binds h_post_attn directly, so a padded row width
            // (weight_cols > hidden) needs the transform to route through
            // scratch.x_buf's permanently-zero tail instead.
            && (scratch.weight_cols == scratch.hidden
                || router_input_transform(moe)
                    == Some(RouterInputTransform::PreExpertsRmsNorm))
    }

    /// Fetch (or build once) the layer's descriptor table. Model swap on
    /// a reused backend is detected by the bank's pointer identity.
    pub(crate) fn descriptor_table_for_layer(
        &self,
        layer_idx: usize,
        moe: &MoeLayerWeights<'_>,
        inter: usize,
        hidden: usize,
    ) -> Option<std::sync::Arc<MoeExpertDescriptorTable>> {
        let bank_ptr = moe.experts_gate_up.first()?.as_ptr() as usize;
        let key = (layer_idx, bank_ptr);
        let mut map = self.moe_descriptor_tables.lock().unwrap();
        if let Some(t) = map.get(&key) {
            return Some(t.clone());
        }
        let table = std::sync::Arc::new(self.build_expert_descriptor_table(moe, inter, hidden)?);
        map.insert(key, table.clone());
        Some(table)
    }

    /// S1 production encode: the full GPU-dataflow MoE layer, consuming
    /// the GPU-resident `h_post_attn` — the host slice's routing role
    /// ends here. Preconditions were checked by
    /// [`Self::gpu_route_supported`]; this function only encodes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_layer_gpu_route(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        h_post_attn: &Buffer,
        new_h: &Buffer,
        eps: f32,
    ) {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        let hidden = scratch.hidden;
        let num_experts = moe.num_experts;

        // Router-input transform, stated explicitly.
        let x_route =
            match router_input_transform(moe).expect("gpu_route_supported checked the policy") {
                RouterInputTransform::Identity => h_post_attn.clone(),
                RouterInputTransform::PreExpertsRmsNorm => {
                    // Norm INTO the scratch staging buffer: it is weight_cols
                    // wide with a permanently-zero tail (the same invariant
                    // the CPU staging path relies on), so padded row widths
                    // read zeros beyond `hidden` exactly as they do today.
                    // The kernel writes [0..hidden]; the tail is never touched.
                    let normed = scratch.x_buf.clone();
                    let weight_buf = self.bufs.get_f32(moe.pre_experts_norm);
                    let hidden_u = hidden as u32;
                    let norm_offset: f32 = 0.0;
                    enc.set_compute_pipeline_state(&self.norms.rms_norm_pipeline);
                    enc.set_buffer(0, Some(h_post_attn), 0);
                    enc.set_buffer(1, Some(&weight_buf), 0);
                    enc.set_buffer(2, Some(&normed), 0);
                    enc.set_bytes(3, 4, &hidden_u as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                    normed
                }
            };

        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then(|| self.bufs.get_f32(moe.router_per_expert_scale));
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let bias_buf = (!moe.router_bias.is_empty()).then(|| self.bufs.get_f32(moe.router_bias));

        let logits = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_route,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        let (ids_buf, weights_buf) = self.encode_moe_router_select(
            enc,
            &logits,
            pe_scale.as_ref(),
            num_experts,
            moe.top_k,
            renormalize,
        );
        self.encode_experts_and_combine_descriptor_x_buf(
            enc,
            &x_route,
            moe,
            scratch,
            table,
            &ids_buf,
            &weights_buf,
            h_post_attn,
            new_h,
        );
    }
}

#[cfg(test)]
mod tests;

/// Timing-only ablation of the expert tail machinery, for the A-12
/// in-situ decomposition: `LARQL_ABLATE_MOE=bias,act,combine` (any
/// subset). **Numbers are wrong by construction** — the run exists to
/// price a component's wall time, never to produce output — and the
/// switch announces itself once on stderr so no measurement can quietly
/// inherit it.
#[derive(Clone, Copy, Default)]
struct MoeAblation {
    bias: bool,
    act: bool,
    combine: bool,
}

fn moe_ablation() -> MoeAblation {
    static A: std::sync::OnceLock<MoeAblation> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        let Ok(spec) = std::env::var("LARQL_ABLATE_MOE") else {
            return MoeAblation::default();
        };
        let mut a = MoeAblation::default();
        for part in spec.split(',') {
            match part.trim() {
                "bias" => a.bias = true,
                "act" => a.act = true,
                "combine" => a.combine = true,
                "" => {}
                other => eprintln!("[moe] unknown LARQL_ABLATE_MOE part: {other}"),
            }
        }
        if a.bias || a.act || a.combine {
            eprintln!(
                "[moe] ABLATED RUN — skipping{}{}{}; numbers are wrong by construction, timing only",
                if a.bias { " bias" } else { "" },
                if a.act { " act" } else { "" },
                if a.combine { " combine" } else { "" },
            );
        }
        a
    })
}

/// Opt-in for the fused down+combine dispatch (`LARQL_MXFP4_EXPERT_DC=1`).
/// A/B/A/B on gpt-oss read −0.21 then +0.12 ms under battery drift —
/// ambiguous, so the split form stays the default until a rested AC
/// re-run decides. (The in-situ ablation's −0.32 ms for `combine` was
/// mostly a dependency-break artifact: skipping the kernel also freed
/// the next layer from waiting on the down projection.)
fn expert_down_combine_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_MXFP4_EXPERT_DC").as_deref() == Ok("1"))
}

/// Control for the x2 expert GEMV arm: `LARQL_MXFP4_EXPERT_X2=0` keeps
/// the single-row vec kernel. Read once.
fn expert_x2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_MXFP4_EXPERT_X2").as_deref() != Ok("0"))
}

/// Opt-in for the fused gate+up expert dispatch (`LARQL_MXFP4_EXPERT_GU=1`).
/// Measured NULL on gpt-oss (−0.04 ms, within noise): the second GEMV's α
/// is hidden behind the first's tail and 1440 threadgroups already fill
/// the machine — the class-aware α model's prediction. Retained as an arm.
fn expert_gu_fusion_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_MXFP4_EXPERT_GU").as_deref() == Ok("1"))
}
