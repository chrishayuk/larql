//! G6c-1: 52 layers in **one** scheduling domain.
//!
//! The layer primitive is gated (G6b, G6c-0.5). What this proves is that
//! stacking it keeps the hidden state and every layer's KV resident, with
//! the host out of the dependency chain from the single upload to the
//! single readback.
//!
//! | proof | what it establishes |
//! |---|---|
//! | parity | 52 composed layers compute the stack program |
//! | policy independence | span and position are read as separate fields |
//! | KV independence | each layer attends to its own cache |
//! | checkpoints | localisation without reintroducing readbacks |
//!
//! ## Why policy independence needs its own proof
//!
//! Muse-Glimmer's 52 layers are 39 sliding(2048)+RoPE and 13 full+NoPE —
//! and those two fields are **perfectly correlated** in this model. A
//! stack encoded only from that pattern cannot distinguish "reads the
//! span" from "reads the position", so a lowering that consulted one and
//! inferred the other would pass. This fixture therefore includes layers
//! with combinations Glimmer never ships (sliding+NoPE, full+RoPE), which
//! only a lowering reading both fields can reproduce.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{AttnShape, AttnWeights, LoweredPosition};
use larql_compute_metal::lowering::ffn::{FfnShape, FfnWeights};
use larql_compute_metal::lowering::stack::{Checkpoint, LayerLowering, StackScratch};
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_models::quant::nvfp4;

const LAYERS: usize = 52;
const HIDDEN: usize = 128;
const INTER: usize = 256;
const NUM_Q: usize = 4;
const NUM_KV: usize = 1;
const HEAD_DIM: usize = 32;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
const T: usize = 8;
const POS: usize = T - 1;
const EPS: f32 = 1e-5;
const POST_EPS: f32 = 1e-8;
const QK_EPS: f32 = 1e-6;
const OFFSET: f32 = 1.0;
const QSCALE: f32 = 3.87;
const THETA: f64 = 500_000.0;
const WINDOW: usize = 4;
const CHECKPOINTS: [usize; 6] = [0, 3, 12, 25, 38, 51];

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(17);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.35
        })
        .collect()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += (*x as f64 - *y as f64).powi(2);
        d += (*x as f64).powi(2);
    }
    (n / d).sqrt()
}

/// This layer's judged policy. Glimmer ships only the first two; the
/// others break the span/position correlation so the encoder must read
/// both fields.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Policy {
    window: Option<usize>,
    rope: bool,
}

fn policy_for(layer: usize) -> Policy {
    match layer {
        // Two layers Glimmer never ships, placed early so any confusion
        // propagates through the rest of the stack.
        1 => Policy {
            window: Some(WINDOW),
            rope: false,
        }, // sliding + NoPE
        2 => Policy {
            window: None,
            rope: true,
        }, // full + RoPE
        // Glimmer's own 3:1 pattern everywhere else.
        l if l % 4 == 3 => Policy {
            window: None,
            rope: false,
        },
        _ => Policy {
            window: Some(WINDOW),
            rope: true,
        },
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
    fn_: Vec<f32>,
    fp: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

fn build_layer(l: u32) -> LayerW {
    let mk = |n, k, s| {
        let f = det(n * k, s);
        (
            nvfp4::quantize(&f, n, k).unwrap(),
            nvfp4::round_trip(&f, n, k).unwrap(),
        )
    };
    let (q, qq) = mk(Q_ROWS, HIDDEN, l * 31 + 1);
    let (k, kq) = mk(KV_ROWS, HIDDEN, l * 31 + 2);
    let (v, vq) = mk(KV_ROWS, HIDDEN, l * 31 + 3);
    let (o, oq) = mk(HIDDEN, Q_ROWS, l * 31 + 4);
    let (g, gq) = mk(Q_ROWS, HIDDEN, l * 31 + 5);
    let (fg, fgq) = mk(INTER, HIDDEN, l * 31 + 6);
    let (fu, fuq) = mk(INTER, HIDDEN, l * 31 + 7);
    let (fd, fdq) = mk(HIDDEN, INTER, l * 31 + 8);
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
        an: det(HIDDEN, l * 31 + 9),
        ap: det(HIDDEN, l * 31 + 10),
        fn_: det(HIDDEN, l * 31 + 11),
        fp: det(HIDDEN, l * 31 + 12),
        // Each layer's own cache: sharing one would make every layer
        // attend to the last layer's keys, which is exactly the defect
        // the KV-independence proof looks for.
        k_cache: det(T * KV_ROWS, l * 31 + 13),
        v_cache: det(T * KV_ROWS, l * 31 + 14),
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

fn rope(v: &mut [f32], heads: usize) {
    let half = HEAD_DIM / 2;
    for h in 0..heads {
        let off = h * HEAD_DIM;
        for i in 0..half {
            let a = POS as f64 * THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64);
            let (s, c) = (a.sin() as f32, a.cos() as f32);
            let (x0, x1) = (v[off + i], v[off + half + i]);
            v[off + i] = x0 * c - x1 * s;
            v[off + half + i] = x0 * s + x1 * c;
        }
    }
}

/// The whole stack on the CPU, layer by layer, in plan order.
fn cpu_stack(h0: &[f32], ws: &[LayerW]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut h = h0.to_vec();
    let mut caps = Vec::new();
    for (l, w) in ws.iter().enumerate() {
        let p = policy_for(l);
        // attention
        let normed = rms(&h, &w.an, EPS, OFFSET);
        let mut q = matvec(&w.qq, &normed, Q_ROWS, HIDDEN);
        let mut k = matvec(&w.kq, &normed, KV_ROWS, HIDDEN);
        let v = matvec(&w.vq, &normed, KV_ROWS, HIDDEN);
        rms_heads(&mut q, NUM_Q);
        q.iter_mut().for_each(|x| *x *= QSCALE);
        if p.rope {
            rope(&mut q, NUM_Q);
        }
        rms_heads(&mut k, NUM_KV);
        if p.rope {
            rope(&mut k, NUM_KV);
        }
        let mut kc = w.k_cache.clone();
        let mut vc = w.v_cache.clone();
        kc[POS * KV_ROWS..].copy_from_slice(&k);
        vc[POS * KV_ROWS..].copy_from_slice(&v);

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let t0 = p.window.filter(|w| T > *w).map(|w| T - w).unwrap_or(0);
        let mut concat = vec![0.0f32; Q_ROWS];
        for head in 0..NUM_Q {
            let kv = head / (NUM_Q / NUM_KV);
            let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let sc: Vec<f32> = (t0..T)
                .map(|t| {
                    let kh = &kc[t * KV_ROWS + kv * HEAD_DIM..t * KV_ROWS + (kv + 1) * HEAD_DIM];
                    qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let ex: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let den: f32 = ex.iter().sum();
            for d in 0..HEAD_DIM {
                concat[head * HEAD_DIM + d] = (t0..T)
                    .enumerate()
                    .map(|(i, t)| ex[i] / den * vc[t * KV_ROWS + kv * HEAD_DIM + d])
                    .sum();
            }
        }
        let gate = matvec(&w.gq, &normed, Q_ROWS, HIDDEN);
        for (c, gv) in concat.iter_mut().zip(&gate) {
            *c *= 1.0 / (1.0 + (-gv).exp());
        }
        let ao = matvec(&w.oq, &concat, HIDDEN, Q_ROWS);
        let ao = rms(&ao, &w.ap, POST_EPS, OFFSET);
        let h1: Vec<f32> = h.iter().zip(&ao).map(|(a, b)| a + b).collect();

        // ffn
        let fnm = rms(&h1, &w.fn_, EPS, OFFSET);
        let g = matvec(&w.fgq, &fnm, INTER, HIDDEN);
        let u = matvec(&w.fuq, &fnm, INTER, HIDDEN);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
            .collect();
        let d = matvec(&w.fdq, &act, HIDDEN, INTER);
        let d = rms(&d, &w.fp, POST_EPS, OFFSET);
        h = h1.iter().zip(&d).map(|(a, b)| a + b).collect();

        if CHECKPOINTS.contains(&l) {
            caps.push(h.clone());
        }
    }
    (h, caps)
}

#[test]
fn fifty_two_layers_lower_into_one_scheduling_domain() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let ws: Vec<LayerW> = (0..LAYERS).map(|l| build_layer(l as u32)).collect();
    let h0 = det(HIDDEN, 999);
    let (want, want_caps) = cpu_stack(&h0, &ws);

    // ── device residency: upload once ───────────────────────────────
    let h_in = gpu.lowering_upload(&h0).unwrap();
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64) as f32)
        .collect();
    let inv_freq_buf = gpu.lowering_upload(&inv_freq).unwrap();
    let sc: Vec<metal::Buffer> = (0..14)
        .map(|i| match i {
            3..=5 => gpu.lowering_scratch(Q_ROWS),
            9..=11 => gpu.lowering_scratch(INTER),
            _ => gpu.lowering_scratch(HIDDEN),
        })
        .collect();
    let scratch = StackScratch {
        h_a: &sc[0],
        h_b: &sc[1],
        attn_normed: &sc[2],
        q: &sc[3],
        gate: &sc[4],
        concat: &sc[5],
        gated: &sc[12],
        attn_out: &sc[6],
        attn_post: &sc[7],
        ffn_normed: &sc[8],
        ffn_gate: &sc[9],
        ffn_up: &sc[10],
        ffn_act: &sc[11],
        ffn_down: &sc[13],
        ffn_post: &sc[2],
        inv_freq: &inv_freq_buf,
    };
    let cap_bufs: Vec<metal::Buffer> = CHECKPOINTS
        .iter()
        .map(|_| gpu.lowering_scratch(HIDDEN))
        .collect();

    // Weight and KV buffers live for the whole stack.
    let mut keep: Vec<metal::Buffer> = Vec::new();
    let mut kv: Vec<(metal::Buffer, metal::Buffer)> = Vec::new();
    for w in &ws {
        kv.push((
            gpu.lowering_upload(&w.k_cache).unwrap(),
            gpu.lowering_upload(&w.v_cache).unwrap(),
        ));
    }
    let wbuf = |m: &nvfp4::Nvfp4Matrix, keep: &mut Vec<metal::Buffer>| {
        keep.push(gpu.lowering_weight(&m.packed));
        keep.push(gpu.lowering_weight(&m.scales));
        (keep.len() - 2, keep.len() - 1)
    };
    let idx: Vec<[(usize, usize); 8]> = ws
        .iter()
        .map(|w| {
            [
                wbuf(&w.q, &mut keep),
                wbuf(&w.k, &mut keep),
                wbuf(&w.v, &mut keep),
                wbuf(&w.o, &mut keep),
                wbuf(&w.g, &mut keep),
                wbuf(&w.fg, &mut keep),
                wbuf(&w.fu, &mut keep),
                wbuf(&w.fd, &mut keep),
            ]
        })
        .collect();
    let norms: Vec<[metal::Buffer; 4]> = ws
        .iter()
        .map(|w| {
            [
                gpu.lowering_upload(&w.an).unwrap(),
                gpu.lowering_upload(&w.ap).unwrap(),
                gpu.lowering_upload(&w.fn_).unwrap(),
                gpu.lowering_upload(&w.fp).unwrap(),
            ]
        })
        .collect();
    let post_a = gpu.lowering_scratch(HIDDEN);
    let post_f = gpu.lowering_scratch(HIDDEN);

    let layers: Vec<LayerLowering> = (0..LAYERS)
        .map(|l| {
            let p = policy_for(l);
            let w = &ws[l];
            let i = &idx[l];
            let pr = |(a, b): (usize, usize), ts: f32| LoweredMatrix::Nvfp4 {
                packed: &keep[a],
                scales: &keep[b],
                tensor_scale: ts,
            };
            LayerLowering {
                attn: AttnWeights {
                    q: pr(i[0], w.q.tensor_scale),
                    k: pr(i[1], w.k.tensor_scale),
                    v: pr(i[2], w.v.tensor_scale),
                    o: pr(i[3], w.o.tensor_scale),
                    gate: Some(pr(i[4], w.g.tensor_scale)),
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    o_bias: None,
                    sinks: None,
                    norm_weight: &norms[l][0],
                    post_norm: Some(PostNorm {
                        weight: &norms[l][1],
                        eps: POST_EPS,
                        weight_offset: OFFSET,
                        scratch: &post_a,
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
                    position: if p.rope {
                        LoweredPosition::Rope { theta: THETA }
                    } else {
                        LoweredPosition::None
                    },
                    window: p.window,
                    softcap: None,
                    position_index: POS,
                    kv_len: T,
                },
                ffn: FfnWeights {
                    gate: LoweredMatrix::Nvfp4 {
                        packed: &keep[i[5].0],
                        scales: &keep[i[5].1],
                        tensor_scale: w.fg.tensor_scale,
                    },
                    up: LoweredMatrix::Nvfp4 {
                        packed: &keep[i[6].0],
                        scales: &keep[i[6].1],
                        tensor_scale: w.fu.tensor_scale,
                    },
                    down: LoweredMatrix::Nvfp4 {
                        packed: &keep[i[7].0],
                        scales: &keep[i[7].1],
                        tensor_scale: w.fd.tensor_scale,
                    },
                    norm_weight: &norms[l][2],
                    post_norm: Some(PostNorm {
                        weight: &norms[l][3],
                        eps: POST_EPS,
                        weight_offset: OFFSET,
                        scratch: &post_f,
                    }),
                },
                ffn_shape: FfnShape {
                    hidden: HIDDEN,
                    intermediate: INTER,
                    norm_eps: EPS,
                    norm_weight_offset: OFFSET,
                },
                k_cache: &kv[l].0,
                v_cache: &kv[l].1,
            }
        })
        .collect();

    let cps: Vec<Checkpoint> = CHECKPOINTS
        .iter()
        .zip(&cap_bufs)
        .map(|(l, b)| Checkpoint {
            after_layer: *l,
            into: b,
        })
        .collect();

    // ── ONE command buffer, ONE wait ────────────────────────────────
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let final_buf = gpu.encode_stack(enc, &h_in, &layers, &scratch, &cps);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got = gpu.lowering_readback(final_buf, HIDDEN).unwrap();

    assert!(
        got.iter().all(|v| v.is_finite()),
        "stack produced non-finite output"
    );
    let parity = rel_rms(&want, &got);
    eprintln!("52-layer stack parity: rel_rms {parity:.3e}");
    assert!(parity < 1e-3, "52-layer stack disagrees: {parity:.3e}");

    // ── checkpoints, all read AFTER the stream completed ────────────
    for ((l, buf), want_cap) in CHECKPOINTS.iter().zip(&cap_bufs).zip(&want_caps) {
        let cap = gpu.lowering_readback(buf, HIDDEN).unwrap();
        let e = rel_rms(want_cap, &cap);
        eprintln!("  checkpoint after layer {l:>2}: rel_rms {e:.3e}");
        assert!(e < 1e-3, "checkpoint after layer {l} diverges: {e:.3e}");
    }
}
