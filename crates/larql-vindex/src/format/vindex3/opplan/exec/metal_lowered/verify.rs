//! VERIFY-N: several positions through the lowered stack in ONE command
//! buffer, each weight stream read once for the block, plus the
//! continuation primitive that keeps only a verified prefix.
//!
//! ```text
//! mark = session.position()
//! ids  = session.verify(&block)      // block[i] fed at mark + i
//! k    = accepted prefix of the proposal against ids
//! session.truncate(mark + k + 1)     // keep block[0..=k]; drop the rest
//! ```
//!
//! `ids[i]` is the greedy id after `block[..=i]` — exactly what `step`
//! would have returned had the block been stepped one token at a time.
//! Truncation needs no cache surgery: attention reads `kv_len` rows and a
//! later position overwrites its slot (the slot-idempotence gate in
//! `test_lowering_kv_evolution.rs`), so moving the position back IS the
//! rewind. That holds for every softmax layer this lowering accepts —
//! sliding layers included, since the cache stores every row and the
//! window is applied at read time. A recurrent layer would need its state
//! restored and is refused before a session exists.

use crate::error::VindexError;
use larql_compute_metal::lowering::head::{
    argmax_partials, ArgmaxScratch, HeadScratch, HeadShape, HeadWeights,
};
use larql_compute_metal::lowering::profile::{SingleEncoder, StageEncoders, StageProfiler};
use larql_compute_metal::lowering::stack::{LayerLowering, StackScratch};
use larql_compute_metal::lowering::DeviceBuffer;

use super::routed::FfnResident;
use super::step::{read_u32, write_f32};
use super::{LoweredSession, PROFILE_MAX_STAGE_RUNS};

/// Stack slots, in `LoweredSession::scratch` order (`step.rs::prepare`).
const SLOT_H_A: usize = 0;
const SLOT_H_B: usize = 1;
const SLOT_ATTN_NORMED: usize = 2;
const SLOT_Q: usize = 3;
const SLOT_GATE: usize = 4;
const SLOT_CONCAT: usize = 5;
const SLOT_ATTN_OUT: usize = 6;
const SLOT_ATTN_POST: usize = 7;
const SLOT_FFN_NORMED: usize = 8;
const SLOT_FFN_GATE: usize = 9;
const SLOT_FFN_UP: usize = 10;
const SLOT_FFN_ACT: usize = 11;
const SLOT_GATED: usize = 12;
const SLOT_FFN_DOWN: usize = 13;
const SLOT_FFN_POST: usize = 14;
const SLOT_HEAD_NORMED: usize = 15;
const SLOT_LOGITS: usize = 16;
const SLOT_RAW_LOGITS: usize = 17;

/// A verify block's device scratch, `rows` positions deep.
pub(super) struct VerifyScratch {
    rows: usize,
    slots: Vec<DeviceBuffer>,
    h_in: DeviceBuffer,
    partial_vals: DeviceBuffer,
    partial_idx: DeviceBuffer,
    /// One u32 per row.
    ids: Vec<DeviceBuffer>,
    /// SPLITK-1 partials `(o, m/l)`, `rows` positions deep.
    splitk: [DeviceBuffer; 2],
    /// Routed layers' block scratch (rung 5a): `rows × top_k` expert
    /// slots. `None` for a dense stack, or under [`ROUTED_BLOCK_ENV`]`=0`.
    routed_block: Option<larql_compute_metal::MoeBlockScratch>,
}

/// Operator control: `LARQL_ROUTED_BLOCK=0` runs a verify block's routed
/// layers row by row (rung 2) instead of as one grouped expert pass — the
/// A arm of the rung-5 measurement. Read once.
pub const ROUTED_BLOCK_ENV: &str = "LARQL_ROUTED_BLOCK";

fn routed_block_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(ROUTED_BLOCK_ENV).as_deref() != Ok("0"))
}

impl LoweredSession<'_> {
    /// KV capacity in positions.
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The next position `step` or `verify` will write.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Run `tokens` at positions `position()..position() + len` in one
    /// command buffer and return each position's greedy id. Advances the
    /// position by `tokens.len()`; pair with [`Self::truncate`] to keep
    /// only a verified prefix.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>, VindexError> {
        let rows = tokens.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        // A look-ahead step may be in flight or prepared for a position
        // this block is about to write.
        self.quiesce();
        let t0 = self.position;
        if t0 + rows > self.max_positions {
            return Err(VindexError::Parse(format!(
                "verify block [{t0}, {}) exceeds the session's {} positions",
                t0 + rows,
                self.max_positions
            )));
        }
        if self.final_norm.is_none() || self.head.is_none() {
            return Err(VindexError::Parse("verify needs an output head".into()));
        }
        self.ensure_verify_scratch(rows);
        // Taken for the call (the session is mutated below while the
        // scratch is bound); a refused block drops it and the next
        // verify reallocates.
        let v = self.verify.take().expect("allocated above");

        let mut h0 = Vec::with_capacity(rows * self.hidden);
        for &t in tokens {
            h0.extend(self.embed(t)?);
        }
        write_f32(&v.h_in, &h0)?;

        let s = &v.slots;
        let scratch = StackScratch {
            h_a: &s[SLOT_H_A],
            h_b: &s[SLOT_H_B],
            attn_normed: &s[SLOT_ATTN_NORMED],
            q: &s[SLOT_Q],
            gate: &s[SLOT_GATE],
            concat: &s[SLOT_CONCAT],
            gated: &s[SLOT_GATED],
            attn_out: &s[SLOT_ATTN_OUT],
            attn_post: &s[SLOT_ATTN_POST],
            ffn_normed: &s[SLOT_FFN_NORMED],
            ffn_gate: &s[SLOT_FFN_GATE],
            ffn_up: &s[SLOT_FFN_UP],
            ffn_act: &s[SLOT_FFN_ACT],
            ffn_down: &s[SLOT_FFN_DOWN],
            ffn_post: &s[SLOT_FFN_POST],
            hybrid: None,
            splitk: Some(larql_compute_metal::ops::kv_splitk::SplitKScratch {
                o_part: &v.splitk[0],
                ml_part: &v.splitk[1],
                rows: v.rows,
                max_q_heads: self.splitk_widths.0,
                max_q_rows: self.splitk_widths.1,
            }),
            routed_block: v.routed_block.as_ref(),
        };
        let layers: Vec<LayerLowering> = self
            .plan
            .layers
            .iter()
            .zip(&self.layers)
            .map(|(plan_layer, r)| self.layer_lowering(plan_layer, r, t0))
            .collect();

        let cmd = self.gpu.new_lowering_command_buffer();
        // Production: one encoder for the block. Under `--profile`: one
        // sampled encoder per stage run, the block recorded into its own
        // ledger (a block is not a token).
        let mut profiler = match self.verify_ledger {
            Some(_) => Some(
                StageProfiler::try_new(&self.gpu.device_ref(), cmd.clone(), PROFILE_MAX_STAGE_RUNS)
                    .map_err(|why| VindexError::Parse(format!("--profile: {why}")))?,
            ),
            None => None,
        };
        let single = profiler
            .is_none()
            .then(|| cmd.new_compute_command_encoder());
        let mut single_encs = single.map(SingleEncoder);
        let encs: &mut dyn StageEncoders = match (profiler.as_mut(), single_encs.as_mut()) {
            (Some(p), _) => p,
            (None, Some(s)) => s,
            (None, None) => unreachable!("one of profiler / single encoder exists"),
        };
        let encoded = self.encode_verify_block(encs, &v, &layers, &scratch, rows);
        // A refusal must still close the encoder: Metal aborts on an
        // encoder released open.
        let samples = match profiler {
            Some(p) => Some(p.finish().1),
            None => {
                single
                    .expect("no profiler means the single encoder")
                    .end_encoding();
                None
            }
        };
        encoded.map_err(|why| VindexError::Parse(format!("verify: {why}")))?;
        cmd.commit();
        self.submissions += 1;
        larql_compute_metal::cb_status::wait_checked(
            &cmd,
            "crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/verify.rs:verify",
        )
        .map_err(|detail| VindexError::Parse(format!("metal-lowered verify refused: {detail}")))?;
        self.last_gpu_ms = larql_compute_metal::lowering::profile::gpu_span_ms(&cmd);
        if let (Some(samples), Some(ledger)) = (samples, self.verify_ledger.as_mut()) {
            if let Some(block) = samples.resolve() {
                ledger.record(&block, self.last_gpu_ms);
            }
        }
        let ids = v
            .ids
            .iter()
            .take(rows)
            .map(read_u32)
            .collect::<Result<Vec<_>, _>>()?;
        self.position = t0 + rows;
        self.last_device_id = ids.last().copied();
        self.verify = Some(v);
        Ok(ids)
    }

    /// The block's GPU work into `encs`: the stack, the head, one argmax
    /// per row. Encodes only; the caller closes, commits and waits.
    fn encode_verify_block(
        &self,
        encs: &mut dyn StageEncoders,
        v: &VerifyScratch,
        layers: &[LayerLowering<'_>],
        scratch: &StackScratch<'_>,
        rows: usize,
    ) -> Result<(), String> {
        let (Some((nw, eps, off)), Some(head)) = (&self.final_norm, &self.head) else {
            return Err("verify needs an output head".into());
        };
        let s = &v.slots;
        let h_final = self
            .gpu
            .encode_stack_rows(encs, &v.h_in, layers, scratch, rows)?;
        let shape = HeadShape {
            hidden: self.hidden,
            vocab: self.vocab,
            norm_eps: *eps,
            norm_weight_offset: *off,
            multiplier: self.head_multiplier,
            softcap: self.head_softcap,
        };
        self.gpu.encode_head_rows(
            encs,
            h_final,
            &s[SLOT_LOGITS],
            &HeadWeights {
                projection: head.as_lowered(),
                norm_weight: nw,
            },
            &HeadScratch {
                normed: &s[SLOT_HEAD_NORMED],
                raw_logits: &s[SLOT_RAW_LOGITS],
            },
            &shape,
            rows,
        )?;
        let row_bytes = (self.vocab * std::mem::size_of::<f32>()) as u64;
        for (i, id) in v.ids.iter().take(rows).enumerate() {
            let enc = encs.stage(larql_compute_metal::lowering::profile::Stage::Head);
            self.gpu.encode_argmax_at(
                enc,
                &s[SLOT_LOGITS],
                i as u64 * row_bytes,
                self.vocab,
                &ArgmaxScratch {
                    partial_vals: &v.partial_vals,
                    partial_idx: &v.partial_idx,
                    out: id,
                },
            );
        }
        Ok(())
    }

    /// Continue as if only positions `..position` had ever been written:
    /// the next `step`/`verify` writes `position`. Moving backwards is the
    /// whole rewind (see the module docs); moving forwards is refused,
    /// since those rows were never computed.
    pub fn truncate(&mut self, position: usize) -> Result<(), VindexError> {
        self.quiesce();
        if position > self.position {
            return Err(VindexError::Parse(format!(
                "truncate to {position} is past the written prefix ({})",
                self.position
            )));
        }
        self.position = position;
        // The device argmax word belongs to a dropped position.
        self.last_device_id = None;
        Ok(())
    }

    /// Allocate (or widen) the verify scratch to `rows` positions.
    fn ensure_verify_scratch(&mut self, rows: usize) {
        if self.verify.as_ref().is_some_and(|v| v.rows >= rows) {
            return;
        }
        let gpu = self.gpu;
        let slots = self
            .scratch_widths
            .iter()
            .map(|w| gpu.lowering_scratch(w * rows))
            .collect();
        let parts = argmax_partials(self.vocab.max(1));
        let (splitk_o, splitk_ml) = larql_compute_metal::ops::kv_splitk::SplitKScratch::lens(
            rows,
            self.splitk_widths.0,
            self.splitk_widths.1,
        );
        self.verify = Some(VerifyScratch {
            rows,
            slots,
            h_in: gpu.lowering_scratch(rows * self.hidden),
            partial_vals: gpu.lowering_scratch(parts),
            partial_idx: gpu.lowering_scratch(parts),
            ids: (0..rows).map(|_| gpu.lowering_scratch(1)).collect(),
            splitk: [
                gpu.lowering_scratch(splitk_o),
                gpu.lowering_scratch(splitk_ml),
            ],
            // Every routed layer of a plan shares its expert shape; one that
            // does not is refused when its block encodes (`fits`).
            routed_block: self
                .layers
                .iter()
                .find_map(|l| match &l.ffn {
                    FfnResident::Routed(r) => Some(&r.scratch),
                    _ => None,
                })
                .filter(|_| routed_block_enabled())
                .map(|layer| larql_compute_metal::MoeBlockScratch::for_layer(gpu, layer, rows)),
        });
    }
}
