//! G6b-1 and G6b-2: the two judged Glimmer semantics the serving path
//! has no kernel for, each gated on its own before attention is
//! assembled from them.
//!
//! Both references are written from the interpreter's rule, not from the
//! shader, and each has a control proving the positive arm could have
//! detected the corresponding defect.

#![cfg(target_os = "macos")]

const HEAD_DIM: usize = 128;
const NUM_HEADS: usize = 32;
const EPS: f32 = 1e-6;

fn deterministic(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(7);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 3.0
        })
        .collect()
}

fn rel_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den).sqrt()
}

/// `rms_norm_heads_no_weight_eps`, transcribed: f64 accumulation, RMS
/// cast to f32, divide.
fn cpu_parameter_free_qk_norm(x: &[f32], num_heads: usize, head_dim: usize, eps: f64) -> Vec<f32> {
    let mut out = x.to_vec();
    for h in 0..num_heads {
        let off = h * head_dim;
        let sq: f64 = (0..head_dim).map(|d| (x[off + d] as f64).powi(2)).sum();
        let rms = (sq / head_dim as f64 + eps).sqrt() as f32;
        for d in 0..head_dim {
            out[off + d] = x[off + d] / rms;
        }
    }
    out
}

#[test]
fn parameter_free_qk_norm_matches_the_interpreter_rule() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let x = deterministic(NUM_HEADS * HEAD_DIM, 11);
    let reference = cpu_parameter_free_qk_norm(&x, NUM_HEADS, HEAD_DIM, EPS as f64);

    let buf = gpu.lowering_upload(&x).expect("upload");
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_parameter_free_qk_norm(enc, &buf, 0, NUM_HEADS, HEAD_DIM, EPS);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got = gpu
        .lowering_readback(&buf, NUM_HEADS * HEAD_DIM)
        .expect("readback");
    gpu.recycle_lowering_scratch(buf);

    let parity = rel_rms(&reference, &got);
    eprintln!("param-free QK norm: rel_rms {parity:.3e}");
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(
        parity < 1e-5,
        "GPU disagrees with the interpreter rule: rel_rms {parity:.3e} \
         (f32 vs f64 accumulation over {HEAD_DIM} terms should be far tighter)"
    );

    // Control: normalising over the whole vector instead of per head is
    // the obvious wrong implementation, and must be detectable.
    let whole = cpu_parameter_free_qk_norm(&x, 1, NUM_HEADS * HEAD_DIM, EPS as f64);
    let control = rel_rms(&whole, &got);
    eprintln!(
        "  control `whole-vector scope`: {:.0}x parity",
        control / parity
    );
    assert!(
        control / parity > 100.0,
        "per-head and whole-vector scope must be distinguishable ({control:.3e} vs {parity:.3e})"
    );
}

#[test]
fn sigmoid_gate_matches_the_judged_semantics() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let n = NUM_HEADS * HEAD_DIM;
    let a = deterministic(n, 21);
    let g = deterministic(n, 22);
    let reference: Vec<f32> = a
        .iter()
        .zip(&g)
        .map(|(av, gv)| av * (1.0 / (1.0 + (-gv).exp())))
        .collect();

    let a_buf = gpu.lowering_upload(&a).expect("upload");
    let g_buf = gpu.lowering_upload(&g).expect("upload");
    let out_buf = gpu.lowering_scratch(n);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_sigmoid_gate(enc, &a_buf, &g_buf, &out_buf, n);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got = gpu.lowering_readback(&out_buf, n).expect("readback");
    for b in [a_buf, g_buf, out_buf] {
        gpu.recycle_lowering_scratch(b);
    }

    let parity = rel_rms(&reference, &got);
    eprintln!("sigmoid gate: rel_rms {parity:.3e}");
    assert!(
        parity < 1e-6,
        "GPU sigmoid gate disagrees: rel_rms {parity:.3e}"
    );

    // Control 1: no gate at all.
    let bypass = rel_rms(&a, &got);
    eprintln!("  control `gate bypassed`: {:.0}x parity", bypass / parity);
    assert!(
        bypass / parity > 100.0,
        "bypassing the gate must be detectable"
    );

    // Control 2: the activation is sigmoid, not tanh or identity — a
    // gate that applied the wrong squashing function would still produce
    // plausible, bounded, wrong numbers.
    let tanh_gate: Vec<f32> = a.iter().zip(&g).map(|(av, gv)| av * gv.tanh()).collect();
    let wrong_act = rel_rms(&tanh_gate, &got);
    eprintln!(
        "  control `tanh activation`: {:.0}x parity",
        wrong_act / parity
    );
    assert!(
        wrong_act / parity > 100.0,
        "sigmoid and tanh gating must be distinguishable"
    );
}

/// Wider than one `embed_gather` threadgroup by two strides plus a tail,
/// so the per-thread loop both wraps and ends mid-stride.
const GATHER_HIDDEN: usize = 2 * larql_compute_metal::kernels::DISPATCH_TG_MAX_THREADS as usize + 3;
const GATHER_VOCAB: usize = 7;
/// Deliberately not the last row, so an off-by-one id reads a real row.
const GATHER_ID: u32 = 5;
/// Gemma's embedding scale for a hidden width near this one (~sqrt(hidden)).
const GATHER_SCALE: f32 = 22.6;

/// `embed_gather` with the id in a device buffer, as a prior argmax
/// leaves it.
fn gather(
    gpu: &larql_compute_metal::MetalBackend,
    table: &metal::Buffer,
    id: u32,
    scale: f32,
) -> Vec<f32> {
    let idx = gpu.lowering_scratch(1);
    // SAFETY: a live one-word shared buffer the GPU is not using yet.
    unsafe { *(idx.contents() as *mut u32) = id };
    let out = gpu.lowering_scratch(GATHER_HIDDEN);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_embed_gather(enc, table, &idx, &out, GATHER_HIDDEN, scale);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    gpu.lowering_readback(&out, GATHER_HIDDEN)
        .expect("readback")
}

#[test]
fn embed_gather_reads_the_device_id_and_applies_only_a_present_scale() {
    // Fail, never skip: a shader that does not compile makes `new()`
    // return None, and a skipped gate reads as a pass.
    let gpu = larql_compute_metal::MetalBackend::new()
        .expect("Metal backend (shader library must compile)");
    let host = deterministic(GATHER_VOCAB * GATHER_HIDDEN, 23);
    let table = gpu.lowering_upload(&host).expect("upload");
    let row = |id: u32| {
        let at = id as usize * GATHER_HIDDEN;
        &host[at..at + GATHER_HIDDEN]
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    // Scaled: one f32 multiply per element on both sides — bit-exact.
    let got = gather(&gpu, &table, GATHER_ID, GATHER_SCALE);
    let want: Vec<f32> = row(GATHER_ID).iter().map(|x| x * GATHER_SCALE).collect();
    assert_eq!(bits(&got), bits(&want), "scaled gather of row {GATHER_ID}");

    // 0.0 encodes an ABSENT multiplier: the row itself, not zeros.
    let unscaled = gather(&gpu, &table, GATHER_ID, 0.0);
    assert_eq!(
        bits(&unscaled),
        bits(row(GATHER_ID)),
        "an absent scale must copy the row unchanged"
    );

    // Control: the id is read from the buffer — a neighbouring id gives
    // a different row, so the equality above could not pass by accident.
    let neighbour = gather(&gpu, &table, GATHER_ID - 1, 0.0);
    assert_eq!(bits(&neighbour), bits(row(GATHER_ID - 1)));
    assert_ne!(bits(&neighbour), bits(&unscaled));
}
