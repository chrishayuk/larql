//! K2 layout tournament — pack the same MXFP4 weights four ways, then race them.
//!
//! The packing is the load-bearing part. All four arms must encode **the same
//! reconstructed values**, or a timing difference could be a work difference.
//! So quantisation happens once, producing `(codes, exponents)`, and both
//! physical layouts are emitted from that single source. Arms B, C and D share
//! a byte-identical artifact and differ only in the kernel's decode, which is
//! what makes `C - B` and `D - B` pure decode effects.
//!
//! ## The spread clamp, and why this is a perf fixture not a parity fixture
//!
//! The interleaved layout carries a 1-bit exponent delta per group, so a
//! superblock's eight exponents must span at most one step. That holds for
//! **97.12% of real K3 superblocks**; here it is *enforced* by clamping, which
//! can alter a value in the rare wide superblock. That is acceptable because
//! this measures throughput and both layouts are emitted from the clamped
//! encoding — but it means these buffers must never be used as a parity oracle.

use crate::shaders::mxfp4_grouped_experts::{ROW_BASE_IDENTITY, ROW_STRIDE_IDENTITY};
use crate::MetalBackend;

/// Weights per e8m0 group.
pub const GROUP: usize = 32;
/// Groups per interleaved superblock.
pub const GROUPS_PER_SB: usize = 8;
/// Packed bytes per group.
pub const GROUP_BYTES: usize = 16;
/// Interleaved superblock: base + delta bitmap + 8 groups.
pub const SB_BYTES: usize = 2 + GROUPS_PER_SB * GROUP_BYTES;
/// Largest exponent step a 1-bit delta can express.
pub const MAX_DELTA: i32 = 1;

const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Nearest fp4 code for `v` expressed in units of the group scale.
fn encode(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0u8 };
    let a = v.abs();
    let mut best = 0usize;
    let mut err = f32::INFINITY;
    for (i, &m) in LUT.iter().enumerate() {
        let e = (a - m).abs();
        if e < err {
            err = e;
            best = i;
        }
    }
    sign | best as u8
}

/// One row's quantisation: 4-bit codes plus one e8m0 exponent per group.
pub struct Quantised {
    /// Two codes per byte, lo nibble first — `GROUP_BYTES` per group.
    pub codes: Vec<u8>,
    /// One e8m0 byte per group, already clamped to a 1-step spread.
    pub exps: Vec<u8>,
    pub groups: usize,
}

/// Quantise `w` (`rows * k`) to MXFP4, clamping each superblock's spread to 1.
pub fn quantise(w: &[f32], rows: usize, k: usize) -> Quantised {
    let groups_per_row = k / GROUP;
    let groups = rows * groups_per_row;
    let mut exps = vec![0u8; groups];

    // Pass 1: the natural exponent per group — smallest e with amax/2^(e-127) <= 6.
    for g in 0..groups {
        let amax = w[g * GROUP..(g + 1) * GROUP]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        exps[g] = if amax == 0.0 {
            127
        } else {
            let e = (amax / 6.0).log2().ceil() as i32 + 127;
            e.clamp(1, 254) as u8
        };
    }

    // Pass 2: narrow each superblock to `base..=base+1` so a 1-bit delta
    // suffices.
    //
    // The clamp must move exponents UP, not down. Lowering a group's exponent
    // shrinks its scale, so `amax / scale` escapes the fp4 range and every
    // large value saturates at 6 — silent, and it changes the arithmetic the
    // bench is supposed to hold constant. Raising it only costs precision in
    // the small groups, which is the acceptable direction.
    for sb in 0..groups / GROUPS_PER_SB {
        let s = &mut exps[sb * GROUPS_PER_SB..(sb + 1) * GROUPS_PER_SB];
        let base = s.iter().max().unwrap().saturating_sub(MAX_DELTA as u8);
        for e in s.iter_mut() {
            *e = (*e).max(base);
        }
    }

    // Pass 3: encode against the (possibly clamped) exponent, so codes and
    // exponents are consistent and both layouts describe identical values.
    let mut codes = vec![0u8; groups * GROUP_BYTES];
    for g in 0..groups {
        let scale = f32::from_bits((exps[g] as u32) << 23);
        for b in 0..GROUP_BYTES {
            let lo = encode(w[g * GROUP + 2 * b] / scale);
            let hi = encode(w[g * GROUP + 2 * b + 1] / scale);
            codes[g * GROUP_BYTES + b] = lo | (hi << 4);
        }
    }
    Quantised {
        codes,
        exps,
        groups,
    }
}

impl Quantised {
    /// Arm A: two streams, as the checkpoint stores it. 4.25 bpw.
    pub fn split(&self) -> (Vec<u8>, Vec<u8>) {
        (self.codes.clone(), self.exps.clone())
    }

    /// Arms B/C/D: one stream, `[base, deltas, 8 x 16 bytes]`. 4.0625 bpw.
    pub fn interleaved(&self) -> Vec<u8> {
        let sbs = self.groups / GROUPS_PER_SB;
        let mut out = Vec::with_capacity(sbs * SB_BYTES);
        for sb in 0..sbs {
            let g0 = sb * GROUPS_PER_SB;
            let e = &self.exps[g0..g0 + GROUPS_PER_SB];
            let base = *e.iter().min().unwrap();
            let mut deltas = 0u8;
            for (i, &ei) in e.iter().enumerate() {
                if ei > base {
                    deltas |= 1 << i;
                }
            }
            out.push(base);
            out.push(deltas);
            for i in 0..GROUPS_PER_SB {
                let g = g0 + i;
                out.extend_from_slice(&self.codes[g * GROUP_BYTES..(g + 1) * GROUP_BYTES]);
            }
        }
        out
    }
}

/// Dot products for the first `rows` rows of one expert, straight from the
/// quantised encoding — the oracle both layouts must reproduce.
pub fn reference(q: &Quantised, x: &[f32], k: usize, rows: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for g in 0..gpr {
                let gi = r * gpr + g;
                let scale = f32::from_bits((q.exps[gi] as u32) << 23) as f64;
                let mut part = 0.0f64;
                for b in 0..GROUP_BYTES {
                    let byte = q.codes[gi * GROUP_BYTES + b];
                    let sv = |c: u8| {
                        let m = LUT[(c & 7) as usize] as f64;
                        if c & 8 != 0 {
                            -m
                        } else {
                            m
                        }
                    };
                    part += sv(byte & 0x0F) * x[g * GROUP + 2 * b] as f64;
                    part += sv(byte >> 4) * x[g * GROUP + 2 * b + 1] as f64;
                }
                acc += scale * part;
            }
            acc as f32
        })
        .collect()
}

/// One arm's measurement.
#[derive(Debug, Clone)]
pub struct ArmResult {
    pub name: &'static str,
    pub bpw: f64,
    pub bytes: f64,
    pub ms: f64,
    pub gbs: f64,
    /// Largest disagreement with the CPU oracle, relative to the oracle's scale.
    /// Covers only the first rows of slot 0 — cheap, but narrow.
    pub max_abs_diff: f32,
    /// Largest relative disagreement with arm A over the FULL output, and the
    /// first index at which it appears. The narrow oracle cannot see addressing
    /// bugs that only bite at high slot or row indices; this does.
    pub cross_diff: f32,
    pub first_bad: Option<usize>,
    /// False for ceiling probes, which compute something other than the matvec.
    pub checked: bool,
}

impl ArmResult {
    pub fn eta(&self, roofline_gbs: f64) -> f64 {
        self.gbs / roofline_gbs
    }
}

fn synth(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 31 + seed * 7919) % 251) as f32) - 125.0) * 0.001)
        .collect()
}

/// Race all four arms at the K3 expert shape with grouped dispatch.
///
/// Protocol matches `profile_grouped_experts`: every arm batched into one
/// command buffer so per-dispatch overhead is amortised rather than measured,
/// and inherently cold — 16 distinct expert payloads far exceed the 16 MB L2.
pub fn race(
    n: usize,
    k: usize,
    top_k: usize,
    batch: usize,
    warmup: usize,
    iters: usize,
) -> Vec<ArmResult> {
    use metal::MTLSize;

    let metal = MetalBackend::new().expect("Metal backend required");

    let (mut split_w, mut split_s, mut inter) = (Vec::new(), Vec::new(), Vec::new());
    let (mut off_split, mut off_inter) = (Vec::<u32>::new(), Vec::<u32>::new());
    // The scale stream gets its own offset table rather than `off_split / 16`.
    // Here the two happen to stay proportional — both streams are appended in
    // lockstep — but the arm A kernel no longer assumes that, and a bench that
    // kept deriving the offset would be exercising a contract the kernel has
    // stopped making.
    let mut off_split_s = Vec::<u32>::new();
    let mut probes: Vec<Quantised> = Vec::new();
    for e in 0..top_k {
        let q = quantise(&synth(n * k, e), n, k);
        let (c, s) = q.split();
        off_split.push(split_w.len() as u32);
        off_split_s.push(split_s.len() as u32);
        split_w.extend_from_slice(&c);
        split_s.extend_from_slice(&s);
        off_inter.push(inter.len() as u32);
        inter.extend_from_slice(&q.interleaved());
        if e < 2 {
            probes.push(q);
        }
    }

    let weights = (n * k * top_k) as f64;
    let x = synth(k, 99);
    // Oracle over the first rows of slot 0: enough to catch any addressing or
    // scale-decode error without a full CPU matvec.
    // Two slots, not one: a per-slot offset bug is invisible from slot 0 alone,
    // and slot 0 is exactly where every offset formula is trivially correct.
    const ORACLE_ROWS: usize = 8;
    let oracle: Vec<f32> = probes
        .iter()
        .flat_map(|q| reference(q, &x, k, ORACLE_ROWS))
        .collect();

    let bw_split = metal.bufs().uncached_bytes(&split_w);
    let bs_split = metal.bufs().uncached_bytes(&split_s);
    let bw_inter = metal.bufs().uncached_bytes(&inter);
    let obytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|o| o.to_ne_bytes()).collect() };
    // Offset tables are ephemeral, so they must NOT go through the
    // address-keyed weight cache — see `BufferCache::uncached_bytes`.
    let boff_split = metal.bufs().uncached_bytes(&obytes(&off_split));
    let boff_split_s = metal.bufs().uncached_bytes(&obytes(&off_split_s));
    let boff_inter = metal.bufs().uncached_bytes(&obytes(&off_inter));
    let xb = metal.bufs().transient_from_f32(&x);
    let out = metal.bufs().output((top_k * n * 4) as u64);
    let (nv, kv, xs) = (n as u32, k as u32, 0u32);

    // `checked` marks arms that must reproduce the oracle. The last two are
    // ceiling probes that deliberately compute something else, so gating them on
    // numerical agreement would be meaningless.
    #[allow(clippy::type_complexity)]
    let arms: [(&'static str, &crate::kernels::KernelHandle, bool, f64, bool); 8] = [
        (
            "A split / LUT16",
            &metal.quant.mxfp4g_split_lut16_pipeline,
            true,
            4.25,
            true,
        ),
        (
            "A2 split / LUT16 vec4",
            &metal.quant.mxfp4g_split_lut16_vec_pipeline,
            true,
            4.25,
            true,
        ),
        (
            "B inter / LUT16",
            &metal.quant.mxfp4g_inter_lut16_pipeline,
            false,
            4.0625,
            true,
        ),
        (
            "C inter / pair-LUT",
            &metal.quant.mxfp4g_inter_pair_pipeline,
            false,
            4.0625,
            true,
        ),
        (
            "D inter / mag+sign",
            &metal.quant.mxfp4g_inter_magsign_pipeline,
            false,
            4.0625,
            true,
        ),
        (
            "G inter / bit-math",
            &metal.quant.mxfp4g_inter_bits_pipeline,
            false,
            4.0625,
            true,
        ),
        (
            "E probe / affine",
            &metal.quant.mxfp4g_inter_affine_pipeline,
            false,
            4.0625,
            false,
        ),
        (
            "F probe / no-X",
            &metal.quant.mxfp4g_inter_nox_pipeline,
            false,
            4.0625,
            false,
        ),
    ];

    let mut results = Vec::new();
    let mut full_ref: Option<Vec<f32>> = None;

    for (name, handle, is_split, bpw, checked) in arms {
        let tiles = (n as u64).div_ceil(handle.rows_per_tg);
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = std::time::Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..batch {
                enc.set_compute_pipeline_state(&handle.state);
                let p = |slot: u64, v: &u32| {
                    enc.set_bytes(slot, 4, v as *const u32 as *const std::ffi::c_void)
                };
                if is_split {
                    enc.set_buffer(0, Some(&bw_split), 0);
                    enc.set_buffer(1, Some(&boff_split), 0);
                    enc.set_buffer(2, Some(&bs_split), 0);
                    enc.set_buffer(3, Some(&boff_split_s), 0);
                    enc.set_buffer(4, Some(&xb), 0);
                    enc.set_buffer(5, Some(&out), 0);
                    p(6, &nv);
                    p(7, &kv);
                    p(8, &xs);
                    // Whole rows, no fused halves: this bench measures one
                    // matvec per slot, so the walk is the identity.
                    p(9, &ROW_BASE_IDENTITY);
                    p(10, &ROW_STRIDE_IDENTITY);
                } else {
                    enc.set_buffer(0, Some(&bw_inter), 0);
                    enc.set_buffer(1, Some(&boff_inter), 0);
                    enc.set_buffer(2, Some(&xb), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    p(4, &nv);
                    p(5, &kv);
                    p(6, &xs);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(tiles, top_k as u64, 1),
                    MTLSize::new(handle.threads_per_tg, 1, 1),
                );
            }
            enc.end_encoding();
            cmd.commit();
            crate::cb_status::wait_or_abort(
                cmd,
                "crates/larql-compute-metal/src/diag/mxfp4_layout.rs:380",
            );
            if i >= warmup {
                times.push(t.elapsed().as_secs_f64() * 1e3 / batch as f64);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = times[times.len() / 2];
        // Against the CPU oracle, not against another arm: comparing arms to
        // each other cannot tell you which one is wrong.
        let got = crate::buffers::read_buffer_f32(&out, top_k * n);
        let scale = oracle.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let max_abs_diff = if !checked {
            0.0
        } else {
            oracle
                .chunks(ORACLE_ROWS)
                .enumerate()
                .flat_map(|(slot, want)| {
                    want.iter()
                        .enumerate()
                        .map(move |(r, w)| (slot * n + r, *w))
                })
                .fold(0.0f32, |m, (i, w)| m.max((w - got[i]).abs() / scale))
        };
        let (cross_diff, first_bad) = match if checked { &full_ref } else { &None } {
            None => (0.0, None),
            Some(r) => {
                let mut worst = 0.0f32;
                let mut idx = None;
                for (i, (a, b)) in r.iter().zip(&got).enumerate() {
                    let d = (a - b).abs() / scale;
                    if d > worst {
                        worst = d;
                    }
                    if d > 1e-3 && idx.is_none() {
                        idx = Some(i);
                    }
                }
                (worst, idx)
            }
        };
        if checked && full_ref.is_none() {
            full_ref = Some(got.clone());
        }
        let bytes = weights * bpw / 8.0;
        results.push(ArmResult {
            name,
            bpw,
            bytes,
            ms,
            gbs: bytes / 1e6 / ms,
            max_abs_diff,
            cross_diff,
            first_bad,
            checked,
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect()
    }

    #[test]
    fn encode_picks_the_nearest_fp4_value_and_keeps_the_sign() {
        assert_eq!(encode(6.0), 7);
        assert_eq!(encode(-6.0), 15);
        assert_eq!(encode(0.0), 0);
        // 2.6 sits between 2 and 3, nearer 3.
        assert_eq!(encode(2.6) & 7, 5);
        assert_eq!(encode(-2.6), 8 | 5);
    }

    #[test]
    fn the_chosen_exponent_keeps_every_value_inside_the_fp4_range() {
        let w = ramp(GROUP * GROUPS_PER_SB);
        let q = quantise(&w, 1, w.len());
        for g in 0..q.groups {
            let scale = f32::from_bits((q.exps[g] as u32) << 23);
            let amax = w[g * GROUP..(g + 1) * GROUP]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(amax / scale <= 6.0 + 1e-6, "group {g} overflows fp4");
        }
    }

    #[test]
    fn every_superblock_spread_fits_a_one_bit_delta() {
        // The interleaved layout is unrepresentable otherwise, so this is a
        // structural guarantee rather than a property of the fixture.
        let w: Vec<f32> = (0..GROUP * GROUPS_PER_SB * 4)
            .map(|i| ((i % 13) as f32 - 6.0) * (1.0 + i as f32 * 0.5))
            .collect();
        let q = quantise(&w, 1, w.len());
        for sb in 0..q.groups / GROUPS_PER_SB {
            let s = &q.exps[sb * GROUPS_PER_SB..(sb + 1) * GROUPS_PER_SB];
            let spread = *s.iter().max().unwrap() - *s.iter().min().unwrap();
            assert!(spread as i32 <= MAX_DELTA, "sb {sb} spread {spread}");
        }
    }

    #[test]
    fn the_two_layouts_carry_the_same_codes_and_exponents() {
        // The whole comparison rests on this: differing bytes, identical values.
        let w = ramp(GROUP * GROUPS_PER_SB * 3);
        let q = quantise(&w, 1, w.len());
        let (codes, exps) = q.split();
        let inter = q.interleaved();
        for sb in 0..q.groups / GROUPS_PER_SB {
            let hdr = sb * SB_BYTES;
            let base = inter[hdr];
            let deltas = inter[hdr + 1];
            for i in 0..GROUPS_PER_SB {
                let g = sb * GROUPS_PER_SB + i;
                let want = base + ((deltas >> i) & 1);
                assert_eq!(want, exps[g], "group {g} exponent");
                let from_inter = &inter[hdr + 2 + i * GROUP_BYTES..][..GROUP_BYTES];
                assert_eq!(from_inter, &codes[g * GROUP_BYTES..][..GROUP_BYTES]);
            }
        }
    }

    #[test]
    fn the_interleaved_layout_is_smaller_by_exactly_the_scale_saving() {
        let w = ramp(GROUP * GROUPS_PER_SB * 5);
        let q = quantise(&w, 1, w.len());
        let (codes, exps) = q.split();
        let split_bpw = ((codes.len() + exps.len()) * 8) as f64 / w.len() as f64;
        let inter_bpw = (q.interleaved().len() * 8) as f64 / w.len() as f64;
        assert!((split_bpw - 4.25).abs() < 1e-12, "{split_bpw}");
        assert!((inter_bpw - 4.0625).abs() < 1e-12, "{inter_bpw}");
    }

    #[test]
    fn a_zero_group_gets_a_representable_exponent() {
        let q = quantise(&vec![0.0; GROUP * GROUPS_PER_SB], 1, GROUP * GROUPS_PER_SB);
        assert!(q.exps.iter().all(|&e| e != 0 && e != 255), "no sentinels");
        assert!(q.codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn eta_divides_by_the_roofline() {
        let a = ArmResult {
            name: "x",
            bpw: 4.25,
            bytes: 0.0,
            ms: 0.0,
            gbs: 330.0,
            max_abs_diff: 0.0,
            cross_diff: 0.0,
            first_bad: None,
            checked: true,
        };
        assert!((a.eta(367.0) - 0.899).abs() < 0.01);
    }

    /// Drive the tournament end-to-end at minimum params, matching the
    /// `profile_grouped_experts` smoke pattern. Skips on hosts without a
    /// Metal device.
    ///
    /// This is the only test that runs the arms against each other rather
    /// than checking a helper in isolation, so it is what would catch an
    /// arm that stopped agreeing with the oracle.
    #[test]
    fn race_smoke_runs_every_arm_and_they_agree() {
        if MetalBackend::new().is_none() {
            return;
        }
        // K a multiple of the superblock, N not a multiple of the tile so the
        // row-remainder path runs too.
        let arms = race(260, GROUP * GROUPS_PER_SB, 2, 1, 0, 1);
        assert!(!arms.is_empty(), "the tournament must produce arms");

        for a in &arms {
            assert!(!a.name.is_empty());
            assert!(a.bytes > 0.0 && a.bpw > 0.0, "{}: no payload", a.name);
            assert!(a.ms.is_finite() && a.ms > 0.0, "{}: bad time", a.name);
            assert!(a.gbs.is_finite() && a.gbs > 0.0, "{}: bad rate", a.name);
            assert!(a.eta(367.0) > 0.0, "{}: eta", a.name);
        }

        // Ceiling probes compute something other than the matvec, so only the
        // checked arms are required to agree with the oracle.
        let checked: Vec<_> = arms.iter().filter(|a| a.checked).collect();
        assert!(!checked.is_empty(), "at least one arm must be checked");
        for a in checked {
            assert!(
                a.max_abs_diff.is_finite(),
                "{}: oracle diff not finite",
                a.name
            );
            assert!(
                a.first_bad.is_none(),
                "{} disagrees with arm A at index {:?} (cross_diff {})",
                a.name,
                a.first_bad,
                a.cross_diff
            );
        }
    }
}
