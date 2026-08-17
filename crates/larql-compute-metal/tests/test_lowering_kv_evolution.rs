//! G6c-3: multi-position KV evolution, device-resident.
//!
//! The attention, stack and head semantics are each already gated. This
//! rung is about **time and state**:
//!
//! ```text
//! position t
//! → write K/V into every layer's own cache slot
//! → attend over the correctly retained history
//! → prior cache contents survive
//! → advance to t+1, no host KV round-trip
//! ```
//!
//! | proof | what it establishes |
//! |---|---|
//! | evolution parity | 8 positions match a CPU reference at every step |
//! | window expiry | sliding layers stop seeing expired positions; full layers still do |
//! | layer disjointness | layer N's cache is not layer N+1's |
//! | slot idempotence | re-encoding a position overwrites its slot, never appends |
//!
//! ## The expiry control is the interesting one
//!
//! It corrupts a KV slot that the sliding window has already passed, and
//! requires the *sliding* layer's output to be unchanged while the *full*
//! layer's output moves. A single assertion in either direction would be
//! weak: unchanged-everywhere would also happen if the corruption never
//! landed, and changed-everywhere would happen if the window were
//! ignored. Requiring the two to disagree is what pins the behaviour.
//!
//! The fixture crosses the boundary on purpose — window 4 over 8
//! positions — and includes the two span/position combinations Glimmer
//! never ships, so neither field can be inferred from the other.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{AttnShape, AttnWeights, LoweredPosition};
use larql_compute_metal::lowering::ffn::{FfnShape, FfnWeights};
use larql_compute_metal::lowering::stack::{LayerLowering, StackScratch};
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_models::quant::nvfp4;

const LAYERS: usize = 4;
const HIDDEN: usize = 96;
const INTER: usize = 192;
const NUM_Q: usize = 4;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 24;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
const POSITIONS: usize = 8;
const WINDOW: usize = 4;
const EPS: f32 = 1e-5;
const POST_EPS: f32 = 1e-8;
const QK_EPS: f32 = 1e-6;
const OFFSET: f32 = 1.0;
const QSCALE: f32 = 3.87;
const THETA: f64 = 500_000.0;

/// Per-layer policy. L0/L1 are Glimmer's own combinations; L2/L3 are the
/// two it never ships, so span and position cannot be inferred from each
/// other.
fn policy(layer: usize) -> (Option<usize>, bool) {
    match layer {
        0 => (Some(WINDOW), true),  // sliding + RoPE
        1 => (None, false),         // full + NoPE
        2 => (Some(WINDOW), false), // sliding + NoPE
        _ => (None, true),          // full + RoPE
    }
}

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(3);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.4
        })
        .collect()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += (*x as f64 - *y as f64).powi(2);
        d += (*x as f64).powi(2);
    }
    if d == 0.0 {
        n.sqrt()
    } else {
        (n / d).sqrt()
    }
}

struct LayerW {
    q: nvfp4::Nvfp4Matrix,
    k: nvfp4::Nvfp4Matrix,
    v: nvfp4::Nvfp4Matrix,
    o: nvfp4::Nvfp4Matrix,
    g: nvfp4::Nvfp4Matrix,
    fg: nvfp4::Nvfp4Matrix,
    fu: nvfp4::Nvfp4Matrix,
    fd: nvfp4::Nvfp4Matrix,
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    oq: Vec<f32>,
    gq: Vec<f32>,
    fgq: Vec<f32>,
    fuq: Vec<f32>,
    fdq: Vec<f32>,
    an: Vec<f32>,
    ap: Vec<f32>,
    fnw: Vec<f32>,
    fp: Vec<f32>,
}

fn build(l: u32) -> LayerW {
    let mk = |n, k, s| {
        let f = det(n * k, s);
        (
            nvfp4::quantize(&f, n, k).unwrap(),
            nvfp4::round_trip(&f, n, k).unwrap(),
        )
    };
    let (q, qq) = mk(Q_ROWS, HIDDEN, l * 41 + 1);
    let (k, kq) = mk(KV_ROWS, HIDDEN, l * 41 + 2);
    let (v, vq) = mk(KV_ROWS, HIDDEN, l * 41 + 3);
    let (o, oq) = mk(HIDDEN, Q_ROWS, l * 41 + 4);
    let (g, gq) = mk(Q_ROWS, HIDDEN, l * 41 + 5);
    let (fg, fgq) = mk(INTER, HIDDEN, l * 41 + 6);
    let (fu, fuq) = mk(INTER, HIDDEN, l * 41 + 7);
    let (fd, fdq) = mk(HIDDEN, INTER, l * 41 + 8);
    LayerW {
        q,
        k,
        v,
        o,
        g,
        fg,
        fu,
        fd,
        qq,
        kq,
        vq,
        oq,
        gq,
        fgq,
        fuq,
        fdq,
        an: det(HIDDEN, l * 41 + 9),
        ap: det(HIDDEN, l * 41 + 10),
        fnw: det(HIDDEN, l * 41 + 11),
        fp: det(HIDDEN, l * 41 + 12),
    }
}

fn rms(v: &[f32], w: &[f32], eps: f32, off: f32) -> Vec<f32> {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    v.iter()
        .zip(w)
        .map(|(x, wv)| x * inv * (off + wv))
        .collect()
}
fn matvec(m: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
        .collect()
}
fn rms_heads(v: &mut [f32], heads: usize) {
    for h in 0..heads {
        let off = h * HEAD_DIM;
        let sq: f64 = (0..HEAD_DIM).map(|d| (v[off + d] as f64).powi(2)).sum();
        let r = (sq / HEAD_DIM as f64 + QK_EPS as f64).sqrt() as f32;
        for d in 0..HEAD_DIM {
            v[off + d] /= r;
        }
    }
}
fn rope(v: &mut [f32], heads: usize, pos: usize) {
    let half = HEAD_DIM / 2;
    for h in 0..heads {
        let off = h * HEAD_DIM;
        for i in 0..half {
            let a = pos as f64 * THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64);
            let (s, c) = (a.sin() as f32, a.cos() as f32);
            let (x0, x1) = (v[off + i], v[off + half + i]);
            v[off + i] = x0 * c - x1 * s;
            v[off + half + i] = x0 * s + x1 * c;
        }
    }
}

/// CPU evolution: caches persist across positions, exactly as the device
/// buffers do.
fn cpu_evolve(
    ws: &[LayerW],
    inputs: &[Vec<f32>],
    kc: &mut [Vec<f32>],
    vc: &mut [Vec<f32>],
    share_caches: bool,
) -> Vec<Vec<f32>> {
    let mut outs = Vec::new();
    for (t, h0) in inputs.iter().enumerate() {
        let mut h = h0.clone();
        for (l, w) in ws.iter().enumerate() {
            let (window, use_rope) = policy(l);
            let ci = if share_caches { 0 } else { l };
            let normed = rms(&h, &w.an, EPS, OFFSET);
            let mut q = matvec(&w.qq, &normed, Q_ROWS, HIDDEN);
            let mut k = matvec(&w.kq, &normed, KV_ROWS, HIDDEN);
            let v = matvec(&w.vq, &normed, KV_ROWS, HIDDEN);
            rms_heads(&mut q, NUM_Q);
            q.iter_mut().for_each(|x| *x *= QSCALE);
            if use_rope {
                rope(&mut q, NUM_Q, t);
            }
            rms_heads(&mut k, NUM_KV);
            if use_rope {
                rope(&mut k, NUM_KV, t);
            }
            kc[ci][t * KV_ROWS..(t + 1) * KV_ROWS].copy_from_slice(&k);
            vc[ci][t * KV_ROWS..(t + 1) * KV_ROWS].copy_from_slice(&v);

            let len = t + 1;
            let t0 = window.filter(|w| len > *w).map(|w| len - w).unwrap_or(0);
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            let mut concat = vec![0.0f32; Q_ROWS];
            for head in 0..NUM_Q {
                let kv = head / (NUM_Q / NUM_KV);
                let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                let sc: Vec<f32> = (t0..len)
                    .map(|tt| {
                        let kh = &kc[ci]
                            [tt * KV_ROWS + kv * HEAD_DIM..tt * KV_ROWS + (kv + 1) * HEAD_DIM];
                        qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                let ex: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let den: f32 = ex.iter().sum();
                for d in 0..HEAD_DIM {
                    concat[head * HEAD_DIM + d] = (t0..len)
                        .enumerate()
                        .map(|(i, tt)| ex[i] / den * vc[ci][tt * KV_ROWS + kv * HEAD_DIM + d])
                        .sum();
                }
            }
            let gate = matvec(&w.gq, &normed, Q_ROWS, HIDDEN);
            for (c, gv) in concat.iter_mut().zip(&gate) {
                *c *= 1.0 / (1.0 + (-gv).exp());
            }
            let ao = rms(
                &matvec(&w.oq, &concat, HIDDEN, Q_ROWS),
                &w.ap,
                POST_EPS,
                OFFSET,
            );
            let h1: Vec<f32> = h.iter().zip(&ao).map(|(a, b)| a + b).collect();

            let fnm = rms(&h1, &w.fnw, EPS, OFFSET);
            let g = matvec(&w.fgq, &fnm, INTER, HIDDEN);
            let u = matvec(&w.fuq, &fnm, INTER, HIDDEN);
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect();
            let d = rms(
                &matvec(&w.fdq, &act, HIDDEN, INTER),
                &w.fp,
                POST_EPS,
                OFFSET,
            );
            h = h1.iter().zip(&d).map(|(a, b)| a + b).collect();
        }
        outs.push(h);
    }
    outs
}

struct Device<'a> {
    gpu: &'a larql_compute_metal::MetalBackend,
    keep: Vec<metal::Buffer>,
    norms: Vec<[metal::Buffer; 4]>,
    kv: Vec<(metal::Buffer, metal::Buffer)>,
    sc: Vec<metal::Buffer>,
    inv_freq: metal::Buffer,
    post_a: metal::Buffer,
    post_f: metal::Buffer,
}

fn setup<'a>(gpu: &'a larql_compute_metal::MetalBackend, ws: &[LayerW], share: bool) -> Device<'a> {
    let mut keep = Vec::new();
    for w in ws {
        for m in [&w.q, &w.k, &w.v, &w.o, &w.g, &w.fg, &w.fu, &w.fd] {
            keep.push(gpu.lowering_weight(&m.packed));
            keep.push(gpu.lowering_weight(&m.scales));
        }
    }
    let norms = ws
        .iter()
        .map(|w| {
            [
                gpu.lowering_upload(&w.an).unwrap(),
                gpu.lowering_upload(&w.ap).unwrap(),
                gpu.lowering_upload(&w.fnw).unwrap(),
                gpu.lowering_upload(&w.fp).unwrap(),
            ]
        })
        .collect();
    let zero = vec![0.0f32; POSITIONS * KV_ROWS];
    let n_caches = if share { 1 } else { ws.len() };
    let kv: Vec<_> = (0..n_caches)
        .map(|_| {
            (
                gpu.lowering_upload(&zero).unwrap(),
                gpu.lowering_upload(&zero).unwrap(),
            )
        })
        .collect();
    let inv: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64) as f32)
        .collect();
    Device {
        gpu,
        keep,
        norms,
        kv,
        sc: (0..14)
            .map(|i| match i {
                3..=5 | 12 => gpu.lowering_scratch(Q_ROWS),
                9..=11 => gpu.lowering_scratch(INTER),
                _ => gpu.lowering_scratch(HIDDEN),
            })
            .collect(),
        inv_freq: gpu.lowering_upload(&inv).unwrap(),
        post_a: gpu.lowering_scratch(HIDDEN),
        post_f: gpu.lowering_scratch(HIDDEN),
    }
}

/// One position through the resident stack: one command buffer, one wait,
/// caches untouched by the host.
fn step(d: &Device<'_>, ws: &[LayerW], h0: &[f32], t: usize, share: bool) -> Vec<f32> {
    let h_in = d.gpu.lowering_upload(h0).unwrap();
    let scratch = StackScratch {
        h_a: &d.sc[0],
        h_b: &d.sc[1],
        attn_normed: &d.sc[2],
        q: &d.sc[3],
        gate: &d.sc[4],
        concat: &d.sc[5],
        gated: &d.sc[12],
        attn_out: &d.sc[6],
        attn_post: &d.sc[7],
        ffn_normed: &d.sc[8],
        ffn_gate: &d.sc[9],
        ffn_up: &d.sc[10],
        ffn_act: &d.sc[11],
        ffn_down: &d.sc[13],
        ffn_post: &d.sc[2],
        inv_freq: &d.inv_freq,
    };
    let layers: Vec<LayerLowering> = (0..ws.len())
        .map(|l| {
            let (window, use_rope) = policy(l);
            let w = &ws[l];
            let b = l * 16;
            let ci = if share { 0 } else { l };
            let pr = |o: usize, ts: f32| LoweredMatrix::Nvfp4 {
                packed: &d.keep[b + o],
                scales: &d.keep[b + o + 1],
                tensor_scale: ts,
            };
            LayerLowering {
                attn: AttnWeights {
                    q: pr(0, w.q.tensor_scale),
                    k: pr(2, w.k.tensor_scale),
                    v: pr(4, w.v.tensor_scale),
                    o: pr(6, w.o.tensor_scale),
                    gate: Some(pr(8, w.g.tensor_scale)),
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    o_bias: None,
                    sinks: None,
                    norm_weight: &d.norms[l][0],
                    post_norm: Some(PostNorm {
                        weight: &d.norms[l][1],
                        eps: POST_EPS,
                        weight_offset: OFFSET,
                        scratch: &d.post_a,
                    }),
                },
                attn_shape: AttnShape {
                    hidden: HIDDEN,
                    num_q_heads: NUM_Q,
                    num_kv_heads: NUM_KV,
                    head_dim: HEAD_DIM,
                    norm_eps: EPS,
                    norm_weight_offset: OFFSET,
                    qk_norm_eps: QK_EPS,
                    parameter_free_q: true,
                    parameter_free_k: true,
                    query_scale: Some(QSCALE),
                    score_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                    position: if use_rope {
                        LoweredPosition::Rope { theta: THETA }
                    } else {
                        LoweredPosition::None
                    },
                    window,
                    softcap: None,
                    position_index: t,
                    kv_len: t + 1,
                },
                ffn: FfnWeights {
                    gate: LoweredMatrix::Nvfp4 {
                        packed: &d.keep[b + 10],
                        scales: &d.keep[b + 11],
                        tensor_scale: w.fg.tensor_scale,
                    },
                    up: LoweredMatrix::Nvfp4 {
                        packed: &d.keep[b + 12],
                        scales: &d.keep[b + 13],
                        tensor_scale: w.fu.tensor_scale,
                    },
                    down: LoweredMatrix::Nvfp4 {
                        packed: &d.keep[b + 14],
                        scales: &d.keep[b + 15],
                        tensor_scale: w.fd.tensor_scale,
                    },
                    norm_weight: &d.norms[l][2],
                    post_norm: Some(PostNorm {
                        weight: &d.norms[l][3],
                        eps: POST_EPS,
                        weight_offset: OFFSET,
                        scratch: &d.post_f,
                    }),
                },
                ffn_shape: FfnShape {
                    hidden: HIDDEN,
                    intermediate: INTER,
                    norm_eps: EPS,
                    norm_weight_offset: OFFSET,
                },
                k_cache: &d.kv[ci].0,
                v_cache: &d.kv[ci].1,
            }
        })
        .collect();

    let cmd = d.gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let out = d.gpu.encode_stack(enc, &h_in, &layers, &scratch, &[]);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let r = d.gpu.lowering_readback(out, HIDDEN).unwrap();
    d.gpu.recycle_lowering_scratch(h_in);
    r
}

#[test]
fn kv_evolves_across_positions_on_the_device() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let ws: Vec<LayerW> = (0..LAYERS).map(|l| build(l as u32)).collect();
    let inputs: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|t| det(HIDDEN, 700 + t as u32))
        .collect();

    let mut kc: Vec<Vec<f32>> = (0..LAYERS)
        .map(|_| vec![0.0; POSITIONS * KV_ROWS])
        .collect();
    let mut vc: Vec<Vec<f32>> = (0..LAYERS)
        .map(|_| vec![0.0; POSITIONS * KV_ROWS])
        .collect();
    let want = cpu_evolve(&ws, &inputs, &mut kc, &mut vc, false);

    // ── Proof 1: evolution parity at every position ─────────────────
    let d = setup(&gpu, &ws, false);
    let mut worst = 0.0f64;
    for (t, h0) in inputs.iter().enumerate() {
        let got = step(&d, &ws, h0, t, false);
        let e = rel_rms(&want[t], &got);
        eprintln!("  position {t}: rel_rms {e:.3e}");
        assert!(got.iter().all(|v| v.is_finite()), "position {t} non-finite");
        assert!(e < 1e-3, "position {t} diverges: {e:.3e}");
        worst = worst.max(e);
    }
    eprintln!("evolution parity: worst {worst:.3e} over {POSITIONS} positions");

    // ── Proof 2: window expiry. Corrupt position 0's slot, which the
    //    window has passed by t=7, and re-run t=7. Sliding layers must
    //    not notice; full layers must.
    let corrupt = [9.0f32; KV_ROWS];
    let last = POSITIONS - 1;
    for (name, layer, expect_change) in [
        ("sliding layer 0", 0usize, false),
        ("full layer 1", 1usize, true),
    ] {
        let d2 = setup(&gpu, &ws, false);
        let mut kc2: Vec<Vec<f32>> = (0..LAYERS)
            .map(|_| vec![0.0; POSITIONS * KV_ROWS])
            .collect();
        let mut vc2: Vec<Vec<f32>> = (0..LAYERS)
            .map(|_| vec![0.0; POSITIONS * KV_ROWS])
            .collect();
        let base = cpu_evolve(&ws, &inputs, &mut kc2, &mut vc2, false);
        for (t, h0) in inputs.iter().enumerate() {
            step(&d2, &ws, h0, t, false);
        }
        // Overwrite the expired slot in one layer's device cache, then
        // re-run the last position. Nothing else changes.
        let raw = d2.kv[layer].0.contents() as *mut f32;
        // SAFETY: shared-storage buffer, no command buffer in flight.
        unsafe { std::ptr::copy_nonoverlapping(corrupt.as_ptr(), raw, KV_ROWS) };
        let after = step(&d2, &ws, &inputs[last], last, false);
        let moved = rel_rms(&base[last], &after);
        eprintln!("  expiry `{name}`: corrupting position 0 moved the output {moved:.3e}");
        if expect_change {
            assert!(
                moved > 1e-3,
                "a full-attention layer must still see position 0 at t={last}; moved {moved:.3e}"
            );
        } else {
            assert!(
                moved < 1e-5,
                "a sliding(w={WINDOW}) layer must NOT see position 0 at t={last}; moved {moved:.3e}"
            );
        }
    }

    // ── Proof 3: each layer's cache is its own ──────────────────────
    let mut kcs: Vec<Vec<f32>> = vec![vec![0.0; POSITIONS * KV_ROWS]];
    let mut vcs: Vec<Vec<f32>> = vec![vec![0.0; POSITIONS * KV_ROWS]];
    let shared_want = cpu_evolve(&ws, &inputs, &mut kcs, &mut vcs, true);
    let shared_moved = rel_rms(&want[last], &shared_want[last]);
    eprintln!("  disjointness: one shared cache vs per-layer moves {shared_moved:.3e}");
    assert!(
        shared_moved > 1e-3,
        "sharing one KV cache across layers must change the result, or proof 1 \
         cannot tell whether the caches are disjoint"
    );

    // ── Proof 4: re-encoding a position overwrites its slot ─────────
    let d3 = setup(&gpu, &ws, false);
    for (t, h0) in inputs.iter().enumerate().take(4) {
        step(&d3, &ws, h0, t, false);
    }
    let once = step(&d3, &ws, &inputs[4], 4, false);
    let twice = step(&d3, &ws, &inputs[4], 4, false);
    let drift = rel_rms(&once, &twice);
    eprintln!("  idempotence: re-encoding position 4 moves the output {drift:.3e}");
    assert!(
        drift < 1e-6,
        "re-encoding a position must overwrite its slot, not append: {drift:.3e}"
    );
}
