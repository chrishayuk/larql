//! Production-shaped synthetic weights and activations.
//!
//! Shape-faithful and value-approximate, deliberately. The arms here are
//! TIMING arms: the SDOT sequence a row executes does not depend on what
//! the codes contain, so reproducing CPU-5's quantiser exactly would buy
//! nothing a probe can spend. What the values must not do is stray
//! somewhere the hardware treats differently — subnormal scales would put
//! FP-assist traps in the measurement — so the ranges below are realistic
//! rather than arbitrary.
//!
//! Nothing here licenses a quality claim. Whether this representation
//! preserves the model is CPU-5/CPU-6's question and is answered on real
//! prompts against a real container, not on synthetic bytes.

use super::kernel::{ACT_BLOCK, PER_WEIGHT_B16, WEIGHT_BLOCK};

/// Deterministic, seeded, and stated: a probe whose fixture changes
/// between runs cannot be compared to its own previous output.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*, chosen for being three lines and reproducible.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

/// One Q8 projection matrix: `rows x in_dim` codes, one `f32` scale per
/// [`WEIGHT_BLOCK`] elements along the input axis.
pub struct Matrix {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub in_dim: usize,
}

impl Matrix {
    fn new(rows: usize, in_dim: usize, rng: &mut Rng) -> Self {
        assert!(
            in_dim.is_multiple_of(WEIGHT_BLOCK),
            "the probe runs the aligned production shape only, so that no \
             tail path is inside the measurement: in_dim {in_dim} is not a \
             multiple of {WEIGHT_BLOCK}"
        );
        let codes = (0..rows * in_dim)
            .map(|_| (rng.unit() * 127.0) as i8)
            .collect();
        // ~1e-2, the order a Q8 weight scale actually takes: peak/127 for
        // a tensor whose weights sit around 1e-2..1e-1.
        let scales = (0..rows * (in_dim / WEIGHT_BLOCK))
            .map(|_| 0.008 + 0.004 * (rng.unit() + 1.0) * 0.5)
            .collect();
        Self {
            codes,
            scales,
            rows,
            in_dim,
        }
    }

    /// Bytes this matrix occupies, scales included.
    ///
    /// The scales are counted because they are READ. A Q8 rate that
    /// ignored its own metadata would flatter the format by exactly what
    /// the metadata costs.
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }

    /// This matrix's codes and scales for one contiguous row range.
    pub fn slab(&self, start: usize, count: usize) -> (&[i8], &[f32]) {
        let per_row = self.in_dim / WEIGHT_BLOCK;
        (
            &self.codes[start * self.in_dim..(start + count) * self.in_dim],
            &self.scales[start * per_row..(start + count) * per_row],
        )
    }
}

/// A set of matrices sized as a whole so the sweep is DRAM-resident.
///
/// Sized as a WHOLE and not per matrix: one 5120x5120 Q8 matrix is 26.2 MB
/// and would sit largely in the system-level cache on this part, which
/// would make a weight-stationary kernel win against a cache it was never
/// going to face in decode. Streaming a bank several times the SLC is what
/// puts the arm in the regime the 27 GB/token figure describes.
pub struct Bank {
    pub mats: Vec<Matrix>,
}

impl Bank {
    pub fn new(count: usize, rows: usize, in_dim: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        Self {
            mats: (0..count)
                .map(|_| Matrix::new(rows, in_dim, &mut rng))
                .collect(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.mats.iter().map(Matrix::bytes).sum()
    }
}

/// One quantised activation vector: asymmetric int8 at [`ACT_BLOCK`].
pub struct QuantAct {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub mids: Vec<f32>,
}

/// Asymmetric block quantisation: `x_i ~ scale[b] * code_i + mid[b]`.
///
/// Per block, not per tensor, and that is not a tuning choice. CPU-5
/// measured a per-TENSOR int8 activation at rel_rms 4.8e-01 against exact
/// weights — a destroyed activation, because the residual stream's peak is
/// ~30x its RMS at depth and one scale over that leaves a typical element
/// about two bits. The span is part of the representation.
pub fn quantise_asymmetric(x: &[f32]) -> QuantAct {
    let blocks = x.len() / ACT_BLOCK;
    let mut codes = vec![0i8; x.len()];
    let mut scales = Vec::with_capacity(blocks);
    let mut mids = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let span = &x[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
        let lo = span.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = span.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mid = 0.5 * (hi + lo);
        // 255 levels across the block's range; a degenerate block gets a
        // scale of zero and codes of zero, which decodes to `mid` exactly.
        let scale = if hi > lo { (hi - lo) / 255.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, v) in span.iter().enumerate() {
            codes[b * ACT_BLOCK + i] = (((v - mid) * inv).round() as i32).clamp(-128, 127) as i8;
        }
        scales.push(scale);
        mids.push(mid);
    }
    QuantAct {
        codes,
        scales,
        mids,
    }
}

/// `n` distinct activation vectors of `in_dim` elements.
///
/// Heavy-tailed on purpose: a Gaussian body with occasional large
/// outliers, because a residual stream at depth has peak/rms ~30 and a
/// well-behaved fixture would make the asymmetric path look better than
/// it is on the real thing.
pub fn activations(n: usize, in_dim: usize, seed: u64) -> Vec<QuantAct> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let x: Vec<f32> = (0..in_dim)
                .map(|i| {
                    let base = rng.unit() + rng.unit() + rng.unit();
                    if i % 512 == 0 {
                        base * 12.0
                    } else {
                        base
                    }
                })
                .collect();
            quantise_asymmetric(&x)
        })
        .collect()
}

/// Bytes ONE activation vector is re-read for each output row.
///
/// Reported separately from weight bytes and never summed with them. This
/// traffic is L1-class — the whole vector is 5 KB of codes plus 2.5 KB of
/// scales and midpoints — so adding it to a DRAM rate would invent
/// bandwidth that never crossed the fabric. K5's finding was that this
/// class of traffic is a load/store OP THROUGHPUT cost, and it was
/// invisible to a ledger that counted stored model bytes.
pub fn activation_bytes_per_row(in_dim: usize) -> usize {
    let blocks = in_dim / ACT_BLOCK;
    in_dim + blocks * 4 * 2
}

/// `SDOT`s one row issues per activation vector, and the weight-only
/// `SDOT`s it issues once regardless of `N`.
///
/// Returned as a pair because that is exactly the distinction the
/// amortisation claim turns on: the first scales with `N`, the second
/// does not, and a curve that looks good because of the second is a
/// smaller finding than one that looks good because of traffic.
pub fn sdots_per_row(in_dim: usize) -> (usize, usize) {
    let groups = in_dim / WEIGHT_BLOCK;
    (groups * PER_WEIGHT_B16, groups * PER_WEIGHT_B16)
}
