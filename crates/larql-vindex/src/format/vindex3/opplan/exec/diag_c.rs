//! DIAG q17 (exploratory, never merged): one-variable numerical switches for the
//! REFERENCE backend, read once from `LARQL_DIAG_C`.
//!
//! Unset (or empty) = the frozen arithmetic, unchanged. Otherwise a
//! comma-separated list of `key=value`:
//!
//! - `mv=<mode>` every reference projection; or per site `mv_q`, `mv_k`, `mv_v`,
//!   `mv_o`, `mv_gate`, `mv_up`, `mv_down`, `mv_head`; groups `mv_attn` (q,k,v,o),
//!   `mv_ffn` (gate,up,down). Modes: `seq` (frozen: sequential, mul then add),
//!   `fma` (sequential, fused multiply-add), `lanes16` (16 partial sums, the
//!   production bf16 kernel's shape, separate mul/add), `lanes16fma` (the production
//!   bf16 kernel's exact arithmetic on the widened weights), `f64` (f64 accumulator,
//!   one rounding), `blas` (larql-compute `matmul_vec`, the production BlasF32 kernel).
//! - `norm=d` (production rms: f32 rms, divide, multiply) | `norm=f64` (one rounding).
//! - `softmax=d` (larql-compute `softmax_in_place_f32`, production) | `softmax=f64`.
//! - `attndot=f64` (q.k accumulated in f64), `attnwv=f64` (weighted V in f64).
//! - `silu=d` (production `x / (1 + exp(-x))`).
//!
//! An unknown key or value panics: a typo must not silently run the baseline.

use std::sync::OnceLock;

pub const ENV: &str = "LARQL_DIAG_C";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mv {
    #[default]
    Seq,
    Fma,
    Lanes16,
    Lanes16Fma,
    F64,
    Blas,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Alt {
    #[default]
    Frozen,
    D,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    Head,
}

#[derive(Debug, Default, Clone)]
pub struct DiagC {
    pub mv: [Mv; 8],
    pub norm: Alt,
    pub softmax: Alt,
    pub attn_dot_f64: bool,
    pub attn_wv_f64: bool,
    pub silu_d: bool,
}

fn mv_of(v: &str) -> Mv {
    match v {
        "seq" => Mv::Seq,
        "fma" => Mv::Fma,
        "lanes16" => Mv::Lanes16,
        "lanes16fma" => Mv::Lanes16Fma,
        "f64" => Mv::F64,
        "blas" => Mv::Blas,
        other => panic!("{ENV}: unknown matvec mode {other:?}"),
    }
}

fn alt_of(v: &str) -> Alt {
    match v {
        "d" => Alt::D,
        "f64" => Alt::F64,
        other => panic!("{ENV}: unknown mode {other:?}"),
    }
}

pub fn parse(spec: &str) -> DiagC {
    let mut c = DiagC::default();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = item
            .split_once('=')
            .unwrap_or_else(|| panic!("{ENV}: {item:?} is not key=value"));
        let sites: &[Site] = match k {
            "mv" => &[Site::Q, Site::K, Site::V, Site::O, Site::Gate, Site::Up, Site::Down, Site::Head],
            "mv_attn" => &[Site::Q, Site::K, Site::V, Site::O],
            "mv_ffn" => &[Site::Gate, Site::Up, Site::Down],
            "mv_q" => &[Site::Q],
            "mv_k" => &[Site::K],
            "mv_v" => &[Site::V],
            "mv_kv" => &[Site::K, Site::V],
            "mv_o" => &[Site::O],
            "mv_gate" => &[Site::Gate],
            "mv_up" => &[Site::Up],
            "mv_down" => &[Site::Down],
            "mv_head" => &[Site::Head],
            _ => &[],
        };
        if !sites.is_empty() {
            for s in sites {
                c.mv[*s as usize] = mv_of(v);
            }
            continue;
        }
        match (k, v) {
            ("norm", v) => c.norm = alt_of(v),
            ("softmax", v) => c.softmax = alt_of(v),
            ("attndot", "f64") => c.attn_dot_f64 = true,
            ("attnwv", "f64") => c.attn_wv_f64 = true,
            ("silu", "d") => c.silu_d = true,
            _ => panic!("{ENV}: unknown switch {item:?}"),
        }
    }
    c
}

pub fn get() -> &'static DiagC {
    static CELL: OnceLock<DiagC> = OnceLock::new();
    CELL.get_or_init(|| {
        let spec = std::env::var(ENV).unwrap_or_default();
        let c = parse(&spec);
        eprintln!("[{ENV}] spec={spec:?} parsed={c:?}");
        c
    })
}

/// The frozen reference `matvec`, or the site's switched accumulation.
pub fn matvec_at(site: Site, weight: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    let mode = get().mv[site as usize];
    if mode == Mv::Blas {
        return larql_compute::cpu::ops::moe::math::matmul_vec(x, weight, out_dim, in_dim);
    }
    // Rows in parallel (speed only): each row's arithmetic is the serial
    // body below, so `seq` is bit-identical to the frozen `kernels::matvec`
    // (checked end to end against the frozen binary's logits).
    let mut y = vec![0.0f32; out_dim];
    y.par_iter_mut().enumerate().for_each(|(o, y_o)| {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        *y_o = match mode {
            Mv::Seq => {
                let mut acc = 0.0f32;
                for (w, v) in row.iter().zip(x) {
                    acc += w * v;
                }
                acc
            }
            Mv::Fma => {
                let mut acc = 0.0f32;
                for (w, v) in row.iter().zip(x) {
                    acc = w.mul_add(*v, acc);
                }
                acc
            }
            Mv::F64 => {
                let mut acc = 0.0f64;
                for (w, v) in row.iter().zip(x) {
                    acc += (*w as f64) * (*v as f64);
                }
                acc as f32
            }
            Mv::Lanes16 => lanes16(row, x, false),
            Mv::Lanes16Fma => lanes16(row, x, true),
            Mv::Blas => unreachable!(),
        };
    });
    y
}

/// The production `bf16_dot_neon` shape on f32 weights: four 4-lane
/// accumulators over 16-element strides, combined `(a0+a1)+(a2+a3)` then a
/// horizontal add, scalar sequential tail. With `fma` it is that kernel's
/// arithmetic exactly (the weights are the exact widening of the same bf16).
fn lanes16(w: &[f32], x: &[f32], fma: bool) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = x.len().min(w.len());
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0usize;
        while i + 16 <= n {
            let (f0, f1) = (vld1q_f32(wp.add(i)), vld1q_f32(wp.add(i + 4)));
            let (f2, f3) = (vld1q_f32(wp.add(i + 8)), vld1q_f32(wp.add(i + 12)));
            let (x0, x1) = (vld1q_f32(xp.add(i)), vld1q_f32(xp.add(i + 4)));
            let (x2, x3) = (vld1q_f32(xp.add(i + 8)), vld1q_f32(xp.add(i + 12)));
            if fma {
                a0 = vfmaq_f32(a0, f0, x0);
                a1 = vfmaq_f32(a1, f1, x1);
                a2 = vfmaq_f32(a2, f2, x2);
                a3 = vfmaq_f32(a3, f3, x3);
            } else {
                a0 = vaddq_f32(a0, vmulq_f32(f0, x0));
                a1 = vaddq_f32(a1, vmulq_f32(f1, x1));
                a2 = vaddq_f32(a2, vmulq_f32(f2, x2));
                a3 = vaddq_f32(a3, vmulq_f32(f3, x3));
            }
            i += 16;
        }
        let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while i < n {
            acc += *w.get_unchecked(i) * *x.get_unchecked(i);
            i += 1;
        }
        acc
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = fma;
        panic!("{ENV}: lanes16 is implemented for aarch64 only");
    }
}

/// Softmax at the reference attention call site.
pub fn softmax_at(scores: &mut [f32]) {
    match get().softmax {
        Alt::Frozen => super::kernels::softmax(scores),
        Alt::D => larql_compute::attention::softmax::softmax_in_place_f32(scores),
        Alt::F64 => {
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e: Vec<f64> = scores.iter().map(|s| (*s as f64 - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            for (s, e) in scores.iter_mut().zip(&e) {
                *s = (e / sum) as f32;
            }
        }
    }
}
