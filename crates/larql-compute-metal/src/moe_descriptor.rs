//! Expert descriptor table — rung C of the GPU-dataflow routing ladder.
//!
//! Builds, once per layer, the immutable per-expert descriptor table the
//! GPU indexes with rung B's `selected_ids`. After construction, every
//! route-dependent binding downstream is `base buffer + offset the GPU
//! already holds`; the CPU contributes nothing per token.
//!
//! ## Descriptor vs route
//!
//! `GpuExpertDescriptor` describes one STORED expert — payload, exponent
//! stream and bias offsets — independent of whether it ever wins a route
//! slot. Layer-uniform facts (format, fused row layout, dims, slice
//! sizes) live once on [`MoeExpertDescriptorTable`], not duplicated per
//! expert: the storage contract keeps them uniform per layer, so a
//! per-expert copy would only be a chance to drift.
//!
//! ## Complete or refuse
//!
//! [`MetalBackend::build_expert_descriptor_table`] returns `None` for
//! any bank the descriptor cannot describe COMPLETELY — cross-buffer
//! experts, offsets past `u32`, ragged slice lengths, bias tables that
//! don't match the stated dims, unregistered regions. A `None` means the
//! caller stays on the legacy CPU-routed path by its own explicit
//! choice; nothing here quietly reconstructs route-dependent bindings on
//! the host inside the supposedly GPU-dataflow path.

use crate::MetalBackend;
use larql_compute::{MoeExpertScales, MoeLayerWeights};
use metal::Buffer;

/// GPU-side descriptor, mirrored by `MoeExpertDescriptor` in
/// `shaders/moe_descriptor.rs`. Field order and widths must not drift —
/// the shader reads this memory as its own struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuExpertDescriptor {
    /// Bytes into [`MoeExpertDescriptorTable::gate_up_base`].
    pub gate_up_payload_off: u32,
    /// Bytes into [`MoeExpertDescriptorTable::down_base`].
    pub down_payload_off: u32,
    /// Bytes into the gate+up e8m0 base; 0 under inline scales.
    pub gate_up_scale_off: u32,
    /// Bytes into the down e8m0 base; 0 under inline scales.
    pub down_scale_off: u32,
    /// f32 elements into the gate+up bias bank; 0 when the bank is absent.
    pub gate_up_bias_off: u32,
    /// f32 elements into the down bias bank; 0 when the bank is absent.
    pub down_bias_off: u32,
}

/// One layer's expert bank, described for GPU-resident routing: the
/// descriptor array plus the handful of base buffers every offset is
/// relative to, and the layer-uniform facts execution needs.
pub struct MoeExpertDescriptorTable {
    /// `[num_experts]` × [`GpuExpertDescriptor`], GPU-resident.
    pub descs: Buffer,
    /// CPU copy of the same table, for inspection gates and encode-time
    /// assertions; identical content to `descs`.
    pub descs_host: Vec<GpuExpertDescriptor>,
    /// Base buffer all gate+up payload offsets index into.
    pub gate_up_base: Buffer,
    /// Base buffer all down payload offsets index into.
    pub down_base: Buffer,
    /// e8m0 stream bases under split scales; `None` under inline scales.
    pub gate_up_scale_base: Option<Buffer>,
    pub down_scale_base: Option<Buffer>,
    /// Whole-bank bias buffers; `None` when the architecture has none.
    pub gate_up_bias_bank: Option<Buffer>,
    pub down_bias_bank: Option<Buffer>,
    /// Layer uniforms: slice sizes every expert was validated against.
    pub gate_up_expert_bytes: usize,
    pub down_expert_bytes: usize,
    pub num_experts: usize,
    /// Whether every payload offset is 16-byte aligned — the vectorised
    /// split kernel's load precondition, computed once at build. Base
    /// buffers are page-aligned (registered mmapped regions or fresh
    /// Metal allocations), so the per-expert offsets are the whole fact.
    pub payload_offsets_vec16: bool,
}

/// One route's GPU-resident bindings, produced by `moe_descriptor_gather`:
/// the gathered slot descriptors plus the expanded per-slot offset tables
/// shaped exactly as the grouped-expert kernels' `offsets` binding.
pub struct SelectedExpertBindings {
    /// `[n_slots]` × [`GpuExpertDescriptor`].
    pub slot_descs: Buffer,
    /// Gate-half byte offsets per slot (payload offset).
    pub gate0_offs: Buffer,
    /// Up-half byte offsets per slot (payload offset + gate_half_bytes).
    pub gate1_offs: Buffer,
    /// Down payload byte offsets per slot.
    pub down_offs: Buffer,
    /// e8m0 exponent-stream byte offsets per slot (split-scale banks;
    /// zero-filled and never bound under inline scales).
    pub gu_scale_offs: Buffer,
    pub dn_scale_offs: Buffer,
}

/// Host-side readback of one gather: the slot descriptors and expanded
/// per-slot offset tables, for inspection gates.
pub struct GatheredBindingsHost {
    pub descs: Vec<GpuExpertDescriptor>,
    pub gate0: Vec<u32>,
    pub gate1: Vec<u32>,
    pub down: Vec<u32>,
}

/// Resolve every expert's slice against ONE registered base buffer with
/// `u32`-addressable offsets. Any miss returns `None` (the whole table
/// refuses — no partial descriptor is a valid descriptor).
fn resolve_uniform_bank(
    bufs: &crate::buffers::BufferCache,
    slices: &[&[u8]],
    expected_len: usize,
) -> Option<(Buffer, Vec<u32>)> {
    let mut base: Option<Buffer> = None;
    let mut offsets = Vec::with_capacity(slices.len());
    for s in slices {
        if s.len() != expected_len {
            return None;
        }
        let (buf, off) = bufs.resolve_region(s)?;
        let off = u32::try_from(off).ok()?;
        match &base {
            None => base = Some(buf),
            Some(b) if b.gpu_address() == buf.gpu_address() => {}
            Some(_) => return None,
        }
        offsets.push(off);
    }
    Some((base?, offsets))
}

impl MetalBackend {
    /// Build the layer's descriptor table, or refuse.
    ///
    /// `inter` / `hidden` are the layer dims the bias banks are
    /// validated against (`[E, 2·inter]` interleaved gate+up rows,
    /// `[E, hidden]` down rows — the `FusedHalf` contract).
    ///
    /// Refusals (`None`), all of them: empty bank; expert slices of
    /// unequal length; any slice outside a registered region or in a
    /// second base buffer; offsets past `u32`; split-scale streams
    /// absent/ragged/unregistered; bias tables whose length contradicts
    /// the stated dims. Every refusal is total — there is no partially
    /// valid table, because a partially valid table is a route-dependent
    /// CPU decision waiting to happen.
    pub fn build_expert_descriptor_table(
        &self,
        moe: &MoeLayerWeights<'_>,
        inter: usize,
        hidden: usize,
    ) -> Option<MoeExpertDescriptorTable> {
        let num_experts = moe.num_experts;
        if num_experts == 0
            || moe.experts_gate_up.len() != num_experts
            || moe.experts_down.len() != num_experts
        {
            return None;
        }

        let gate_up_expert_bytes = moe.experts_gate_up.first()?.len();
        let down_expert_bytes = moe.experts_down.first()?.len();
        if gate_up_expert_bytes == 0 || down_expert_bytes == 0 {
            return None;
        }
        let (gate_up_base, gu_offs) =
            resolve_uniform_bank(&self.bufs, &moe.experts_gate_up, gate_up_expert_bytes)?;
        let (down_base, dn_offs) =
            resolve_uniform_bank(&self.bufs, &moe.experts_down, down_expert_bytes)?;

        // Split-scale banks: the exponent streams are part of the
        // representation, so a table without them is not a descriptor of
        // this bank at all.
        let (gu_scale, dn_scale) = match &moe.expert_scales {
            MoeExpertScales::Inline => (None, None),
            MoeExpertScales::Paired { gate_up, down } => {
                if gate_up.len() != num_experts || down.len() != num_experts {
                    return None;
                }
                let gu_len = gate_up.first()?.len();
                let dn_len = down.first()?.len();
                if gu_len == 0 || dn_len == 0 {
                    return None;
                }
                let gu = resolve_uniform_bank(&self.bufs, gate_up, gu_len)?;
                let dn = resolve_uniform_bank(&self.bufs, down, dn_len)?;
                (Some(gu), Some(dn))
            }
        };

        // Bias banks are flat per-layer tables; validate against the
        // stated dims and compute element offsets. A non-empty table of
        // the wrong length is a contradiction with the layer dims, not a
        // case to work around.
        let gate_up_bias_row = 2 * inter;
        let gate_up_bias_bank = if moe.experts_gate_up_bias.is_empty() {
            None
        } else {
            if moe.experts_gate_up_bias.len() != num_experts * gate_up_bias_row {
                return None;
            }
            u32::try_from((num_experts - 1) * gate_up_bias_row).ok()?;
            Some(self.bufs.get_f32(moe.experts_gate_up_bias))
        };
        let down_bias_bank = if moe.experts_down_bias.is_empty() {
            None
        } else {
            if moe.experts_down_bias.len() != num_experts * hidden {
                return None;
            }
            u32::try_from((num_experts - 1) * hidden).ok()?;
            Some(self.bufs.get_f32(moe.experts_down_bias))
        };

        let descs_host: Vec<GpuExpertDescriptor> = (0..num_experts)
            .map(|e| GpuExpertDescriptor {
                gate_up_payload_off: gu_offs[e],
                down_payload_off: dn_offs[e],
                gate_up_scale_off: gu_scale.as_ref().map_or(0, |(_, o)| o[e]),
                down_scale_off: dn_scale.as_ref().map_or(0, |(_, o)| o[e]),
                gate_up_bias_off: if gate_up_bias_bank.is_some() {
                    (e * gate_up_bias_row) as u32
                } else {
                    0
                },
                down_bias_off: if down_bias_bank.is_some() {
                    (e * hidden) as u32
                } else {
                    0
                },
            })
            .collect();

        let bytes = unsafe {
            std::slice::from_raw_parts(
                descs_host.as_ptr() as *const u8,
                std::mem::size_of_val(descs_host.as_slice()),
            )
        };
        let descs = self.bufs.transient_from_bytes(bytes);

        let payload_offsets_vec16 = descs_host
            .iter()
            .all(|d| d.gate_up_payload_off % 16 == 0 && d.down_payload_off % 16 == 0);
        if !payload_offsets_vec16
            && self.quant.mxfp4_grouped_arm
                == crate::shaders::mxfp4_grouped_experts::Mxfp4Arm::SplitLut16Vec
        {
            // Once per layer bank, at build — fired evidence that the
            // vectorised arm was demoted, never a silent slow path.
            eprintln!(
                "[moe-descriptor] payload offsets not 16-byte aligned; \
                 vectorised MXFP4 arm demoted to scalar for this bank"
            );
        }

        Some(MoeExpertDescriptorTable {
            descs,
            descs_host,
            gate_up_base,
            down_base,
            gate_up_scale_base: gu_scale.map(|(b, _)| b),
            down_scale_base: dn_scale.map(|(b, _)| b),
            gate_up_bias_bank,
            down_bias_bank,
            gate_up_expert_bytes,
            down_expert_bytes,
            num_experts,
            payload_offsets_vec16,
        })
    }

    /// Encode `out[slot] = descs[selected_ids[slot]]` — the single
    /// runtime indirection of the GPU-dataflow architecture — plus the
    /// expansion into the per-slot offset tables the grouped-expert
    /// kernels already consume (their signature is unchanged; the call
    /// site swaps `set_bytes` for `set_buffer`). `gate_half_bytes` is the
    /// ContiguousHalves layer fact, stated once as a uniform.
    pub(crate) fn encode_descriptor_gather(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        table: &MoeExpertDescriptorTable,
        selected_ids: &Buffer,
        n_slots: usize,
        gate_half_bytes: u32,
    ) -> SelectedExpertBindings {
        let slot_descs = self
            .bufs
            .output((n_slots * std::mem::size_of::<GpuExpertDescriptor>()) as u64);
        let gate0_offs = self.bufs.output((n_slots * 4) as u64);
        let gate1_offs = self.bufs.output((n_slots * 4) as u64);
        let down_offs = self.bufs.output((n_slots * 4) as u64);
        let gu_scale_offs = self.bufs.output((n_slots * 4) as u64);
        let dn_scale_offs = self.bufs.output((n_slots * 4) as u64);
        let n = n_slots as u32;
        enc.set_compute_pipeline_state(&self.moe_descriptor_gather_pipeline);
        enc.set_buffer(0, Some(&table.descs), 0);
        enc.set_buffer(1, Some(selected_ids), 0);
        enc.set_buffer(2, Some(&slot_descs), 0);
        enc.set_buffer(3, Some(&gate0_offs), 0);
        enc.set_buffer(4, Some(&gate1_offs), 0);
        enc.set_buffer(5, Some(&down_offs), 0);
        enc.set_buffer(6, Some(&gu_scale_offs), 0);
        enc.set_buffer(7, Some(&dn_scale_offs), 0);
        enc.set_bytes(8, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(
            9,
            4,
            &gate_half_bytes as *const u32 as *const std::ffi::c_void,
        );
        let num_experts = table.descs_host.len() as u32;
        enc.set_bytes(10, 4, &num_experts as *const u32 as *const std::ffi::c_void);
        enc.set_buffer(11, Some(&self.route_guard.counter), 0);
        enc.dispatch_threads(
            metal::MTLSize::new(n_slots as u64, 1, 1),
            metal::MTLSize::new(n_slots as u64, 1, 1),
        );
        SelectedExpertBindings {
            slot_descs,
            gate0_offs,
            gate1_offs,
            down_offs,
            gu_scale_offs,
            dn_scale_offs,
        }
    }

    /// Encode the GPU replacement for the CPU bias-staging loop:
    /// de-interleave each selected expert's gate/up bias rows from the
    /// layer bank into slot-aligned `[n_slots, inter]` outputs (the
    /// layout the activation kernels already consume).
    pub(crate) fn encode_bias_stage(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        bias_bank: &Buffer,
        slot_descs: &Buffer,
        (gate_out, up_out): (&Buffer, &Buffer),
        inter: usize,
        n_slots: usize,
    ) {
        let inter_u32 = inter as u32;
        let n = n_slots as u32;
        enc.set_compute_pipeline_state(&self.moe_bias_stage_pipeline);
        enc.set_buffer(0, Some(bias_bank), 0);
        enc.set_buffer(1, Some(slot_descs), 0);
        enc.set_buffer(2, Some(gate_out), 0);
        enc.set_buffer(3, Some(up_out), 0);
        enc.set_bytes(4, 4, &inter_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(inter as u64, n_slots as u64, 1),
            metal::MTLSize::new(64.min(inter as u64).max(1), 1, 1),
        );
    }

    /// Round-trip surface for the rung-C gates: gather descriptors for
    /// `selected_ids` on the GPU and read them back. Production keeps the
    /// buffer GPU-side; the readback exists for inspection.
    pub fn descriptor_gather_roundtrip(
        &self,
        table: &MoeExpertDescriptorTable,
        selected_ids: &[u32],
        gate_half_bytes: u32,
    ) -> Option<GatheredBindingsHost> {
        if selected_ids.is_empty()
            || selected_ids
                .iter()
                .any(|&i| i as usize >= table.num_experts)
        {
            return None;
        }
        let n_slots = selected_ids.len();
        let ids_bytes: Vec<u8> = selected_ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        let ids_buf = self.bufs.transient_from_bytes(&ids_bytes);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let out = self.encode_descriptor_gather(enc, table, &ids_buf, n_slots, gate_half_bytes);
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_descriptor.rs:382",
        );

        let ptr = out.slot_descs.contents() as *const GpuExpertDescriptor;
        if ptr.is_null() {
            return None;
        }
        let descs = unsafe { std::slice::from_raw_parts(ptr, n_slots) }.to_vec();
        let read_u32 = |b: &Buffer| -> Option<Vec<u32>> {
            let p = b.contents() as *const u32;
            if p.is_null() {
                return None;
            }
            Some(unsafe { std::slice::from_raw_parts(p, n_slots) }.to_vec())
        };
        Some(GatheredBindingsHost {
            descs,
            gate0: read_u32(&out.gate0_offs)?,
            gate1: read_u32(&out.gate1_offs)?,
            down: read_u32(&out.down_offs)?,
        })
    }

    /// Round-trip surface for the rung-C bias gate: GPU-stage the
    /// selected experts' gate/up bias rows and read back the slot-aligned
    /// outputs `(gate, up)`, each `[n_slots × inter]`. No CPU staging
    /// happens between the bank upload and the readback — the route is
    /// consumed entirely by GPU kernels.
    pub fn bias_stage_roundtrip(
        &self,
        table: &MoeExpertDescriptorTable,
        selected_ids: &[u32],
        inter: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let bank = table.gate_up_bias_bank.as_ref()?;
        if selected_ids.is_empty()
            || selected_ids
                .iter()
                .any(|&i| i as usize >= table.num_experts)
        {
            return None;
        }
        let n_slots = selected_ids.len();
        let ids_bytes: Vec<u8> = selected_ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        let ids_buf = self.bufs.transient_from_bytes(&ids_bytes);
        let gate_out = self.bufs.output((n_slots * inter * 4) as u64);
        let up_out = self.bufs.output((n_slots * inter * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let gate_half = (table.gate_up_expert_bytes / 2) as u32;
        let bindings = self.encode_descriptor_gather(enc, table, &ids_buf, n_slots, gate_half);
        self.encode_bias_stage(
            enc,
            bank,
            &bindings.slot_descs,
            (&gate_out, &up_out),
            inter,
            n_slots,
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/moe_descriptor.rs:443",
        );

        let gate = crate::buffers::try_read_buffer_f32(&gate_out, n_slots * inter)?;
        let up = crate::buffers::try_read_buffer_f32(&up_out, n_slots * inter)?;
        Some((gate, up))
    }
}
