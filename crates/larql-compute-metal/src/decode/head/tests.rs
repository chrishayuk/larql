//! Tests for [`super`] — the decode-time LM head (TOKEN-B1 rung 2).
//!
//! Two things need pinning, and they are different in kind.
//!
//! **The refusals.** Every precondition returns `None`, and a `None` is
//! harmless: the caller runs the unfused head. What would *not* be
//! harmless is a precondition that fails to fire — the head would then
//! read a store at a stride it cannot explain, or apply RMS where the
//! model wants LayerNorm, and produce a plausible distribution over the
//! wrong logits. So each refusal is pinned to its own cause, and the
//! admitting case is pinned alongside it so a blanket `None` (which would
//! pass every refusal test on its own) cannot masquerade as correct.
//!
//! **The parity.** `encode_decode_head` must agree with the unfused path
//! it replaces. The reference here is deliberately the *shipped* one —
//! CPU RMS norm followed by `q4k_matvec_topk`, which rung 1 already pinned
//! against a full readback — so this test measures exactly what rung 2
//! changed: where the work is encoded, not what it computes.

use larql_compute::{DecodeHeadPlan, NormType, QuantMatVec};
use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

use crate::MetalBackend;

const HIDDEN: usize = 256;
const VOCAB: usize = 512;
const EPS: f32 = 1e-6;

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// A plan that every precondition admits, over a store whose geometry is
/// real: `VOCAB` rows of `HIDDEN` elements, quantised as the writer would.
struct Fixture {
    lm_head: Vec<u8>,
    norm_w: Vec<f32>,
    hidden_state: Vec<f32>,
}

fn fixture() -> Fixture {
    let w = synth(VOCAB * HIDDEN, 0xA11CE);
    Fixture {
        lm_head: larql_compute::cpu::ops::q4_common::quantize_q4_k(&w),
        norm_w: synth(HIDDEN, 0xB0B).iter().map(|v| 1.0 + 0.1 * v).collect(),
        hidden_state: synth(HIDDEN, 0xC0FFEE),
    }
}

impl Fixture {
    fn plan(&self) -> DecodeHeadPlan<'_> {
        DecodeHeadPlan {
            norm_type: NormType::RmsNorm,
            final_norm_weight: &self.norm_w,
            norm_eps: EPS,
            norm_offset: 0.0,
            lm_head_q4k: &self.lm_head,
            vocab: VOCAB,
            cols: HIDDEN,
            top_k: 5,
        }
    }
}

/// Encode `plan` against `hidden_state` on its own command buffer and
/// return the reduced top-K. `None` means the head refused.
///
/// The production caller encodes onto the decode's still-open encoder; a
/// dedicated buffer here isolates the head from the decode loop without
/// changing what is encoded.
fn run_head(
    metal: &MetalBackend,
    hidden_state: &[f32],
    plan: &DecodeHeadPlan<'_>,
) -> Option<Vec<(u32, f32)>> {
    let h_buf = metal.bufs.transient_from_f32(hidden_state);
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let out = metal.encode_decode_head(enc, &h_buf, HIDDEN, plan);
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/decode/head/tests.rs:88",
    )
    .expect("command buffer completed");
    out.map(|b| b.reduce_and_recycle(&metal.bufs))
}

/// CPU RMS norm, matching `rms_norm` semantics: scale by the reciprocal
/// root-mean-square, then multiply by `weight + offset`.
fn cpu_rms_norm(x: &[f32], weight: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(v, w)| v * inv * (w + offset))
        .collect()
}

#[test]
fn admits_a_well_formed_plan() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let hits = run_head(&metal, &f.hidden_state, &f.plan());
    let hits = hits.expect("a plan meeting every precondition must be admitted");
    assert_eq!(hits.len(), 5, "top_k entries come back");
    assert!(
        hits.windows(2).all(|w| w[0].1 >= w[1].1),
        "top-K is descending: {hits:?}"
    );
    assert!(
        hits.iter()
            .all(|(id, s)| (*id as usize) < VOCAB && s.is_finite()),
        "ids are in-vocab and scores finite: {hits:?}"
    );
}

/// The rung's actual claim: moving the head inside the command buffer did
/// not change the head. Same hidden state, same store — the fused path and
/// the shipped unfused path must select the same tokens.
#[test]
fn fused_head_matches_the_unfused_reference() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let plan = f.plan();

    let fused = run_head(&metal, &f.hidden_state, &plan).expect("plan admitted");

    // Reference: normalise on the CPU, then the rung-1 fused matvec+top-K.
    let normed = cpu_rms_norm(&f.hidden_state, &f.norm_w, EPS, 0.0);
    let reference = metal
        .q4k_matvec_topk(&f.lm_head, &normed, VOCAB, HIDDEN, 5)
        .expect("reference path available on this device");

    assert_eq!(
        fused.len(),
        reference.len(),
        "both paths return the same number of candidates"
    );
    // Scores are compared with a tolerance because the two paths reduce in
    // a different order, not because they compute different things; the
    // token ids must match exactly.
    for (i, (got, want)) in fused.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            got.0, want.0,
            "rank {i}: fused picked {} where the reference picked {}\n  fused={fused:?}\n  ref={reference:?}",
            got.0, want.0
        );
        assert!(
            (got.1 - want.1).abs() <= 1e-3 * want.1.abs().max(1.0),
            "rank {i}: score {} vs reference {}",
            got.1,
            want.1
        );
    }
}

#[test]
fn refuses_a_norm_it_does_not_implement() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let mut plan = f.plan();
    plan.norm_type = NormType::LayerNorm;
    assert!(
        run_head(&metal, &f.hidden_state, &plan).is_none(),
        "LayerNorm has mean-subtraction and an optional bias; applying RMS \
         instead would yield a plausible distribution over the wrong logits"
    );
}

#[test]
fn refuses_a_norm_weight_of_the_wrong_length() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let short = vec![1.0f32; HIDDEN - 1];
    let mut plan = f.plan();
    plan.final_norm_weight = &short;
    assert!(run_head(&metal, &f.hidden_state, &plan).is_none());
}

#[test]
fn refuses_an_empty_vocab() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let mut plan = f.plan();
    plan.vocab = 0;
    assert!(run_head(&metal, &f.hidden_state, &plan).is_none());
}

#[test]
fn refuses_a_row_narrower_than_the_hidden_state() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let mut plan = f.plan();
    plan.cols = HIDDEN - Q4_K_BLOCK_ELEMS;
    assert!(
        run_head(&metal, &f.hidden_state, &plan).is_none(),
        "a row narrower than hidden cannot consume the whole query"
    );
}

#[test]
fn refuses_a_row_width_that_is_not_whole_super_blocks() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let mut plan = f.plan();
    plan.cols = HIDDEN + 1;
    assert!(
        run_head(&metal, &f.hidden_state, &plan).is_none(),
        "a store that does not divide into super-blocks must not be read at \
         a guessed stride"
    );
}

#[test]
fn refuses_a_byte_length_that_contradicts_the_geometry() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();
    let truncated = &f.lm_head[..f.lm_head.len() - Q4_K_BLOCK_BYTES];
    let mut plan = f.plan();
    plan.lm_head_q4k = truncated;
    assert!(
        run_head(&metal, &f.hidden_state, &plan).is_none(),
        "vocab x row_bytes must equal the store's length exactly"
    );
}

#[test]
fn refuses_a_top_k_outside_one_partial_row() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();

    let mut zero = f.plan();
    zero.top_k = 0;
    assert!(run_head(&metal, &f.hidden_state, &zero).is_none());

    let mut too_wide = f.plan();
    too_wide.top_k = crate::shaders::f32_gemv::K_TOPK + 1;
    assert!(
        run_head(&metal, &f.hidden_state, &too_wide).is_none(),
        "the partial reduction emits K_TOPK per threadgroup and cannot serve more"
    );
}

/// Padding is the case the production model actually hits: gpt-oss stores
/// a 2880-wide hidden at 3072 columns. The pad must contribute nothing, so
/// a padded store must agree with the unpadded one on the same query.
///
/// The test has to *earn* that assertion. `BufferCache::output` hands back
/// freshly-allocated Metal buffers already zeroed, so a version of this
/// test that simply ran the head would pass whether or not the head zeroes
/// the query's tail — it would be pinning Metal's allocator, not the code.
/// So the scratch buffer the head is about to take is deliberately dirtied
/// and recycled first, which is exactly the state steady-state decode
/// reaches after the first token. Verified to fail when the tail-zeroing in
/// `encode_decode_head` is removed.
#[test]
fn zero_padded_rows_do_not_contribute_to_the_logits() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = fixture();

    // Build a store whose rows are one super-block wider, with the tail
    // holding non-zero weights. If the head let the pad through, those
    // weights would move the scores.
    let padded_cols = HIDDEN + Q4_K_BLOCK_ELEMS;
    let base = synth(VOCAB * HIDDEN, 0xA11CE);
    let mut wide = vec![0.0f32; VOCAB * padded_cols];
    for r in 0..VOCAB {
        wide[r * padded_cols..r * padded_cols + HIDDEN]
            .copy_from_slice(&base[r * HIDDEN..(r + 1) * HIDDEN]);
        for c in HIDDEN..padded_cols {
            // Row-DEPENDENT, deliberately: a constant tail would shift every
            // logit by the same amount and leave the ranking intact, so the
            // test would pass whether or not the pad leaked.
            wide[r * padded_cols + c] = ((r % 13) as f32) * 0.5 - 3.0;
        }
    }
    let wide_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wide);

    let unpadded = run_head(&metal, &f.hidden_state, &f.plan()).expect("plan admitted");

    // Poison the pooled buffer the head will pop for `norm_out`: same byte
    // size, non-zero throughout. Without the tail-zeroing these values
    // survive into the matvec's padded columns.
    let poisoned = metal.bufs.output((padded_cols * 4) as u64);
    unsafe {
        let p = poisoned.contents() as *mut f32;
        for i in 0..padded_cols {
            *p.add(i) = 3.0;
        }
    }
    metal.bufs.recycle(poisoned);

    let mut plan = f.plan();
    plan.lm_head_q4k = &wide_q4k;
    plan.cols = padded_cols;
    let padded = run_head(&metal, &f.hidden_state, &plan).expect("padded plan admitted");

    assert_eq!(
        padded.iter().map(|h| h.0).collect::<Vec<_>>(),
        unpadded.iter().map(|h| h.0).collect::<Vec<_>>(),
        "the query's zero tail must null the padded columns\n  padded={padded:?}\n  unpadded={unpadded:?}"
    );
}
