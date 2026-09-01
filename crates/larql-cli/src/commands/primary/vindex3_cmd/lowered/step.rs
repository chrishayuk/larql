//! One token through the lowered session: embed on the host, then the
//! entire stack and head in **one** command buffer with a single wait.
//!
//! ## Encode ahead of the input
//!
//! Everything a token's command buffer binds is determined by its
//! *position* — KV slot, rope index, scratch, weights — except the
//! embedding row, which depends on the sampled id. So the command buffer
//! for position `t+1` is encoded **while the GPU executes position `t`**
//! (`prepare`), and when `t`'s argmax is known only the embedding row is
//! written into the prepared input buffer before commit. The A-12 ledger
//! priced host encode at 0.5–0.8 ms per token (gpt-oss / Gemma 4) sitting
//! on the critical path with the GPU idle; this takes it off.
//!
//! Two invariants keep this honest:
//! - the prepared buffer is committed only after the previous token's
//!   command buffer has completed, so every scratch buffer the two share
//!   is read and written by one command buffer at a time;
//! - a prepared step is bound to one position; if the caller's position
//!   moved (a capture step, a fresh session), it is discarded and the
//!   step is encoded on the spot.
//!
//! Capturing steps (`--dump-layers`) encode on the spot: they bind
//! per-layer checkpoint buffers that are read back immediately.
//!
//! Split from `mod.rs` (which holds residency and construction) so each
//! file carries one concern; the session type is shared, the impl is
//! continued here.

use larql_compute_metal::lowering::head::{ArgmaxScratch, HeadScratch, HeadShape, HeadWeights};
use larql_compute_metal::lowering::profile::{
    gpu_span_ms, SingleEncoder, Stage, StageEncoders, StageProfiler, StageSamples,
};
use larql_compute_metal::lowering::stack::{HybridScratch, LayerLowering, StackScratch};
use larql_compute_metal::lowering::{DeviceBuffer, DeviceCommandBuffer};
use larql_vindex::error::VindexError;

use super::{LoweredSession, HYBRID_SCRATCH_BASE, PROFILE_MAX_STAGE_RUNS};

/// A position's command buffer, encoded but not committed: its input
/// buffer is written with the embedding row once the id is known — or,
/// in gather mode (1c), looked up on the device from the argmax result,
/// in which case the buffer may already be committed.
pub(super) struct PreparedStep {
    position: usize,
    cmd: DeviceCommandBuffer,
    samples: Option<StageSamples>,
    /// 1c: the step's first op gathers the embedding from the device
    /// argmax buffer — no host write needed, and the command buffer is
    /// committed AHEAD of the previous step completing (Metal's hazard
    /// tracking orders the gather after the argmax write).
    gather: bool,
    /// Whether `cmd` has already been committed (gather mode).
    committed: bool,
    /// The hidden-state input the stack reads; dedicated to this step
    /// until it completes.
    h_in: DeviceBuffer,
    /// Whether a head was encoded (logits land in the head's slot).
    has_logits: bool,
    /// Per-layer checkpoint buffers, for capturing steps only.
    captures: Vec<DeviceBuffer>,
    /// Host encode time, ms.
    encode_ms: f64,
}

impl LoweredSession<'_> {
    /// Step one token: embed on the host, then the entire stack, head
    /// and argmax in one command buffer with a single wait. Returns the
    /// argmax id (`None` without a head) — four bytes leave the device,
    /// not the vocabulary; `last_logits` reads the full vector on demand.
    /// Encodes the *next* position's command buffer while this one runs.
    pub fn step(&mut self, token: u32) -> Result<Option<u32>, VindexError> {
        self.step_impl(token, None)
    }

    /// Declare that every following `step` continues from the device
    /// argmax (greedy decode). Look-ahead steps then gather their
    /// embedding on the device and are committed before their
    /// predecessor completes — the host leaves the token loop (1c).
    pub fn begin_decode(&mut self) {
        self.decode_chain = true;
    }

    /// Wait out any committed look-ahead step and discard it. Call
    /// before reading session state (`last_logits`) that an in-flight
    /// step would still be writing. Returns the in-flight step's argmax
    /// id, if it ran to a head — the id belonging to the logits
    /// `last_logits` will now read.
    pub fn quiesce(&mut self) -> Option<u32> {
        let p = self.prepared.take()?;
        let mut id = None;
        if p.committed {
            larql_compute_metal::cb_status::wait_or_abort(
                &p.cmd,
                "crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/step.rs:quiesce",
            );
            if let (Some(am), true) = (&self.argmax, p.has_logits) {
                id = read_u32(&am[2 + p.position % 2]).ok();
                self.last_device_id = id;
            }
            self.position += 1;
        }
        self.discard(p);
        id
    }

    /// The logits of the most recent completed step, read from the
    /// device. Valid until the next step commits; `None` without a head.
    pub fn last_logits(&self) -> Option<Vec<f32>> {
        self.head.as_ref()?;
        self.gpu
            .lowering_readback(&self.scratch[HEAD_LOGITS_SLOT], self.vocab)
    }

    /// One step, capturing the embedding row and every layer's output for
    /// this position — the per-layer planes a `shannon layer-diff` reads.
    /// `layers_out[i]` is layer `i`'s post-FFN-residual hidden state.
    pub fn step_capturing(
        &mut self,
        token: u32,
    ) -> Result<(Option<u32>, Vec<f32>, Vec<Vec<f32>>), VindexError> {
        let mut embedding = Vec::new();
        let mut layers_out = Vec::new();
        let logits = self.step_impl(token, Some((&mut embedding, &mut layers_out)))?;
        Ok((logits, embedding, layers_out))
    }

    /// The embedding row for `token`, scaled and (if the plan judges it)
    /// weightlessly normalised — the one host computation a token needs.
    fn embed(&self, token: u32) -> Result<Vec<f32>, VindexError> {
        let row = &self.embed_table[token as usize * self.hidden..][..self.hidden];
        let embedding = self
            .plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("no embedding".into()))?;
        let mut h0 = row.to_vec();
        if let Some(scale) = embedding.scale {
            h0.iter_mut().for_each(|v| *v *= scale);
        }
        // The judged embedding norm: Muse-Glimmer RMS-normalises every
        // looked-up row **weightlessly**. Nothing in the checkpoint
        // records that it happens — there is no operand to classify, so
        // no closure or parity gate over the container can see it, and
        // omitting it produced entirely plausible logits with the wrong
        // argmax (368 against the oracle's 13796). It was caught here
        // only by comparing against the independent model oracle, which
        // is precisely why that anchor exists.
        if let Some(norm) = embedding.norm {
            let ms = h0.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / h0.len() as f64;
            let inv = 1.0 / (ms + norm.eps).sqrt();
            h0.iter_mut().for_each(|v| *v = (*v as f64 * inv) as f32);
        }
        Ok(h0)
    }

    fn step_impl(
        &mut self,
        token: u32,
        capture: Option<(&mut Vec<f32>, &mut Vec<Vec<f32>>)>,
    ) -> Result<Option<u32>, VindexError> {
        let t = self.position;
        // A prepared buffer is for exactly one position; a gather-mode
        // buffer is only valid if the caller's token IS the id the device
        // argmax produced (greedy decode) — teacher forcing a different
        // id discards it. Captures always encode fresh.
        let prepared = match self.prepared.take() {
            Some(p)
                if p.position == t
                    && capture.is_none()
                    && (!p.gather || self.last_device_id == Some(token)) =>
            {
                p
            }
            Some(stale) => {
                self.discard(stale);
                self.prepare(t, capture.is_some(), false)?
            }
            None => self.prepare(t, capture.is_some(), false)?,
        };
        let mut h0 = Vec::new();
        if !prepared.gather {
            // The one input the encode could not know: the embedding row,
            // written into the bound buffer before commit.
            h0 = self.embed(token)?;
            write_f32(&prepared.h_in, &h0)?;
        }
        self.last_encode_ms = prepared.encode_ms;
        if !prepared.committed {
            prepared.cmd.commit();
        }

        // Overlap: encode the next position while this one executes. Only
        // within the session's KV capacity, and never for a capturing
        // step (its caller may be dumping a fixed prompt). When this step
        // carries a device argmax, the next step gathers its embedding
        // from it on the device and is COMMITTED now, before this step
        // completes — the queue pipelines and the host leaves the token
        // loop entirely (1c).
        if capture.is_none() && t + 1 < self.max_positions {
            let gather_next = self.decode_chain
                && self.device_embed.is_some()
                && self.argmax.is_some()
                && self.ledger.is_none()
                && gather_enabled();
            let next = self.prepare(t + 1, false, gather_next)?;
            if next.gather {
                next.cmd.commit();
            }
            self.prepared = Some(PreparedStep {
                committed: gather_next,
                ..next
            });
        }

        // A step whose buffer did not complete has no logits and no
        // argmax; the refusal carries Metal's own description up through
        // the step's error channel instead of reading stale scratch.
        larql_compute_metal::cb_status::wait_checked(
            &prepared.cmd,
            "crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/step.rs:step",
        )
        .map_err(|detail| VindexError::Parse(format!("metal-lowered step refused: {detail}")))?;
        self.last_gpu_ms = gpu_span_ms(&prepared.cmd);
        if let (Some(samples), Some(ledger)) = (&prepared.samples, self.ledger.as_mut()) {
            if let Some(token) = samples.resolve() {
                ledger.record(&token, self.last_gpu_ms);
            }
        }

        let out = match (&self.argmax, prepared.has_logits) {
            (Some(am), true) => Some(read_u32(&am[2 + t % 2])?),
            // Control arm: no device argmax — read the vocabulary back and
            // scan it on the host, as before.
            (None, true) => self.last_logits().map(|l| host_argmax(&l)),
            _ => None,
        };
        self.last_device_id = out;
        if let Some((embedding, layers_out)) = capture {
            *embedding = h0;
            for buf in &prepared.captures {
                layers_out.push(
                    self.gpu
                        .lowering_readback(buf, self.hidden)
                        .ok_or_else(|| VindexError::Parse("capture readback failed".into()))?,
                );
            }
        }
        self.discard(prepared);
        self.position += 1;
        Ok(out)
    }

    /// Return a prepared step's pooled buffers. For a committed step this
    /// must follow its completion; for an uncommitted one nothing ever
    /// read them.
    pub(super) fn discard(&self, p: PreparedStep) {
        // A committed step may still be executing; its buffers cannot
        // rejoin the pool until the GPU is done with them.
        if p.committed {
            larql_compute_metal::cb_status::wait_or_abort(
                &p.cmd,
                "crates/larql-cli/src/commands/primary/vindex3_cmd/lowered/step.rs:discard",
            );
        }
        for buf in p.captures {
            self.gpu.recycle_lowering_scratch(buf);
        }
        self.gpu.recycle_lowering_scratch(p.h_in);
    }

    /// Encode position `t`'s whole command buffer — stack and head —
    /// against an input buffer whose contents are written later (host
    /// mode) or gathered from the device argmax as the buffer's first op
    /// (gather mode, 1c).
    fn prepare(
        &self,
        t: usize,
        capturing: bool,
        gather: bool,
    ) -> Result<PreparedStep, VindexError> {
        let encode_started = std::time::Instant::now();
        // Per-layer capture buffers (hidden-sized), read back after the
        // command buffer completes — a copy inside the stream, never a
        // mid-stream readback.
        let captures: Vec<DeviceBuffer> = if capturing {
            (0..self.plan.layers.len())
                .map(|_| self.gpu.lowering_scratch(self.hidden))
                .collect()
        } else {
            Vec::new()
        };
        let h_in = self.gpu.lowering_scratch(self.hidden);

        let s = &self.scratch;
        let scratch = StackScratch {
            h_a: &s[0],
            h_b: &s[1],
            attn_normed: &s[2],
            q: &s[3],
            gate: &s[4],
            concat: &s[5],
            gated: &s[12],
            attn_out: &s[6],
            attn_post: &s[7],
            ffn_normed: &s[8],
            ffn_gate: &s[9],
            ffn_up: &s[10],
            ffn_act: &s[11],
            ffn_down: &s[13],
            ffn_post: &s[14],
            hybrid: (s.len() > HYBRID_SCRATCH_BASE).then(|| HybridScratch {
                dense_out: &s[HYBRID_SCRATCH_BASE],
                router_in: &s[HYBRID_SCRATCH_BASE + 1],
                expert_sum: &s[HYBRID_SCRATCH_BASE + 2],
                experts_out: &s[HYBRID_SCRATCH_BASE + 3],
                branch_sum: &s[HYBRID_SCRATCH_BASE + 4],
                zero: &s[HYBRID_SCRATCH_BASE + 6],
            }),
        };

        let layers: Vec<LayerLowering> = self
            .plan
            .layers
            .iter()
            .zip(&self.layers)
            .map(|(plan_layer, r)| self.layer_lowering(plan_layer, r, t))
            .collect();

        let checkpoints: Vec<larql_compute_metal::lowering::stack::Checkpoint> = captures
            .iter()
            .enumerate()
            .map(
                |(i, buf)| larql_compute_metal::lowering::stack::Checkpoint {
                    after_layer: i,
                    into: buf,
                },
            )
            .collect();
        let cmd = self.gpu.new_lowering_command_buffer();
        // Production: one encoder for the token. Profiling: one sampled
        // encoder per stage run, same dispatches in the same order.
        let mut profiler = if self.ledger.is_some() {
            Some(
                StageProfiler::try_new(&self.gpu.device_ref(), cmd.clone(), PROFILE_MAX_STAGE_RUNS)
                    .map_err(|why| VindexError::Parse(format!("--profile: {why}")))?,
            )
        } else {
            None
        };
        // Only one encoder may be open on a command buffer: the single
        // encoder exists only when the profiler does not.
        let single = profiler
            .is_none()
            .then(|| cmd.new_compute_command_encoder());
        let mut single_encs = single.map(SingleEncoder);
        let encs: &mut dyn StageEncoders = match (profiler.as_mut(), single_encs.as_mut()) {
            (Some(p), _) => p,
            (None, Some(s)) => s,
            (None, None) => unreachable!("one of profiler / single encoder exists"),
        };
        if gather {
            let (table, am) = match (&self.device_embed, &self.argmax) {
                (Some(tb), Some(am)) => (tb, am),
                _ => {
                    return Err(VindexError::Parse(
                        "gather prepare without device state".into(),
                    ))
                }
            };
            // The PREVIOUS position's argmax word (parity-alternated:
            // this step is `t`, its input id came from `t-1`).
            let idx = &am[2 + (t + 1) % 2];
            let scale = self
                .plan
                .embedding
                .as_ref()
                .and_then(|e| e.scale)
                .unwrap_or(0.0);
            let enc = encs.stage(Stage::AttnNorm);
            self.gpu
                .encode_embed_gather(enc, table, idx, &h_in, self.hidden, scale);
        }
        let h_final = self
            .gpu
            .encode_stack(encs, &h_in, &layers, &scratch, &checkpoints);

        let has_logits = match (&self.final_norm, &self.head) {
            (Some((nw, eps, off)), Some(head)) => {
                let hs = HeadScratch {
                    normed: &s[15],
                    raw_logits: &s[17],
                };
                let hw = HeadWeights {
                    projection: head.as_lowered(),
                    norm_weight: nw,
                };
                let shape = HeadShape {
                    hidden: self.hidden,
                    vocab: self.vocab,
                    norm_eps: *eps,
                    norm_weight_offset: *off,
                    multiplier: self.head_multiplier,
                    softcap: self.head_softcap,
                };
                self.gpu
                    .encode_head(encs, h_final, &s[HEAD_LOGITS_SLOT], &hw, &hs, &shape);
                if let Some([vals, idx, out_even, out_odd]) = &self.argmax {
                    let enc = encs.stage(Stage::Head);
                    self.gpu.encode_argmax(
                        enc,
                        &s[HEAD_LOGITS_SLOT],
                        self.vocab,
                        &ArgmaxScratch {
                            partial_vals: vals,
                            partial_idx: idx,
                            out: if t.is_multiple_of(2) {
                                out_even
                            } else {
                                out_odd
                            },
                        },
                    );
                }
                true
            }
            _ => false,
        };
        if let Some(enc) = single {
            enc.end_encoding();
        }
        let samples = profiler.map(|p| p.finish().1);
        Ok(PreparedStep {
            position: t,
            cmd,
            samples,
            gather,
            committed: false,
            h_in,
            has_logits,
            captures,
            encode_ms: encode_started.elapsed().as_secs_f64() * 1e3,
        })
    }
}

/// Scratch slot the head writes its final logits to (slot 15 is the
/// normed hidden, 17 the raw logits).
const HEAD_LOGITS_SLOT: usize = 16;

/// Control for the 1c GPU-directed decode chain:
/// `LARQL_LOWERED_GATHER=0` keeps the host embed + per-step commit.
fn gather_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_LOWERED_GATHER").as_deref() != Ok("0"))
}

/// Host argmax: strict `>` scanning upward, first maximum on ties — the
/// contract the device kernel reproduces.
pub(super) fn host_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, v)| {
                if *v > bv {
                    (i, *v)
                } else {
                    (bi, bv)
                }
            },
        )
        .0 as u32
}

/// Read the first `u32` of a shared device buffer.
fn read_u32(buf: &DeviceBuffer) -> Result<u32, VindexError> {
    let ptr = buf.contents() as *const u32;
    if ptr.is_null() || (buf.length() as usize) < std::mem::size_of::<u32>() {
        return Err(VindexError::Parse("argmax readback failed".into()));
    }
    // SAFETY: the buffer holds at least one u32 (checked) and the
    // command buffer that wrote it has completed.
    Ok(unsafe { std::ptr::read_volatile(ptr) })
}

/// Write `values` into a shared device buffer's contents.
fn write_f32(buf: &DeviceBuffer, values: &[f32]) -> Result<(), VindexError> {
    let ptr = buf.contents() as *mut f32;
    if ptr.is_null() || (buf.length() as usize) < std::mem::size_of_val(values) {
        return Err(VindexError::Parse("hidden upload failed".into()));
    }
    // SAFETY: the buffer is at least `values.len() * 4` bytes (checked)
    // and no committed command buffer reads it until after this write.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
    Ok(())
}
