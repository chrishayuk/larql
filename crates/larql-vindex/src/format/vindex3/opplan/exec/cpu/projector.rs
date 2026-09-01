//! What a dense projection kernel must expose, and who threads it.

/// Who owns the machine while a primitive runs.
///
/// Declared by the kernel because only the kernel knows: a BLAS call has
/// already made its own arrangements, a hand-written SIMD loop has not,
/// and a literal reference transcription must stay single-threaded to
/// remain readable beside the source it transcribes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuParallelism {
    /// One call, one thread. The reference oracle.
    Serial,
    /// The implementation is already threaded internally — Accelerate's
    /// `sgemv`. The executor calls it ONCE over the whole output and does
    /// not partition: measured at 1.14x for the effort, and slower than
    /// serial on some shapes.
    LibraryOwned,
    /// The executor partitions output rows across its workers. Kernels
    /// that scale — the fused BF16 kernel went 34.3 -> 122.0 GB/s from
    /// one worker to twelve — and every future low-bit kernel.
    ExternalPool,
}

/// A dense projection `y = W x`, expressed as a row-range primitive.
///
/// Deliberately NOT `fn project(w, x) -> Vec<f32>`: that shape forces the
/// kernel to decide its own threading, which is the thing this design
/// forbids. `project_rows` computes a contiguous slice of the output from
/// exactly the weight rows that produce it, so the executor is free to
/// cut the work however measurement says it should.
pub trait DenseProjector: Sync {
    /// Who threads this kernel. See [`CpuParallelism`].
    fn parallelism(&self) -> CpuParallelism;

    /// Compute `out.len()` output rows.
    ///
    /// `weight_rows` is row-major and covers exactly `out.len()` rows of
    /// `x.len()` columns — the executor has already sliced it. The kernel
    /// must not look outside it, and must not spawn.
    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]);

    /// Whether this kernel serves `n` activations from ONE weight
    /// traversal at this geometry.
    ///
    /// Exists so a caller can tell a weight-stationary measurement from a
    /// silently looped one. CPU-5's K1 spent two rungs reading a fallback
    /// as a result; a schedule that cannot announce itself is a schedule
    /// nobody can adjudicate. Default `false`, which is the honest answer
    /// for every kernel that has not implemented one.
    fn is_weight_stationary(&self, _weight: WeightRows<'_>, _in_dim: usize, _n: usize) -> bool {
        false
    }

    /// The same rows against `n` activations.
    ///
    /// `out` is `rows * n`, POSITION-MINOR: row `r`'s `n` results are
    /// contiguous at `out[r*n .. (r+1)*n]`, because the weight row is
    /// what a stationary kernel keeps resident.
    ///
    /// The default is the loop it replaces, so every existing kernel
    /// keeps working untouched and only one with something better
    /// overrides it. Overriding it is a claim about SCHEDULE, never about
    /// arithmetic: each activation must keep its own codes, scales,
    /// midpoints and accumulator, and no sum may cross positions.
    fn project_rows_many(
        &self,
        weight_rows: WeightRows<'_>,
        xs: &[&[f32]],
        out: &mut [f32],
        n: usize,
    ) {
        project_rows_looped(self, weight_rows, xs, out, n);
    }
}

/// One traversal per activation, interleaved into the position-minor
/// layout. The default body, and the fallback an overriding kernel takes
/// when its own preconditions do not hold — shared so the two cannot
/// drift into computing different things.
pub fn project_rows_looped(
    kernel: &(impl DenseProjector + ?Sized),
    weight_rows: WeightRows<'_>,
    xs: &[&[f32]],
    out: &mut [f32],
    n: usize,
) {
    let rows = out.len() / n;
    let mut scratch = vec![0.0f32; rows];
    for (i, x) in xs.iter().enumerate().take(n) {
        kernel.project_rows(weight_rows, x, &mut scratch);
        for (r, v) in scratch.iter().enumerate() {
            out[r * n + i] = *v;
        }
    }
}

/// A contiguous slab of weight rows, in whatever representation it is
/// resident as.
///
/// The representation stays COMPACT to this point. CPU-1B's decisive
/// finding was that widening to a scratch matrix first costs more than
/// the traffic it saves — 27.3 GB/s against 122.0 — so a compact format
/// must reach the kernel still compact and be decoded into registers.
#[derive(Clone, Copy, Debug)]
pub enum WeightRows<'a> {
    F32(&'a [f32]),
    /// Symmetric int8 with one f32 scale per `block` elements along the
    /// INPUT axis: `w[r][i] = codes[r][i] * scales[r][i / block]`.
    ///
    /// Blocked rather than per row because the scale is what a row's
    /// dynamic range costs everyone in it — one outlier in 5120 weights
    /// would flatten the rest to a handful of levels. Blocked along the
    /// input axis specifically so a kernel accumulates a block, scales
    /// once, and moves on, instead of scaling every element.
    ///
    /// Unlike [`Self::Bf16`] this is LOSSY: it changes the values the
    /// checkpoint stores. The kernel below only has to consume the format
    /// faithfully; whether the format is good enough is a separate
    /// question with its own gates.
    Q8 {
        codes: &'a [i8],
        scales: &'a [f32],
        /// Per-`SUM_BLOCK` code sums — the asymmetric path's execution
        /// index. Empty where no arm consumes it.
        sums: &'a [i16],
        block: usize,
    },
    /// Symmetric int4, two codes per byte, one f32 scale per `block`
    /// elements.
    ///
    /// **Byte `j` of a block holds elements `j` and `j + block/2`**, not
    /// `2j` and `2j+1`. Adjacent packing would make one 16-byte load
    /// yield 32 INTERLEAVED elements, and every kernel would spend its
    /// time undoing that; half-block packing yields two contiguous runs
    /// that pair directly with two runs of the activation.
    Q4 {
        packed: &'a [u8],
        scales: &'a [f32],
        block: usize,
    },
    /// Big-endian-agnostic bf16 code units: each is the top 16 bits of the
    /// f32 it denotes, so widening is `(bits as u32) << 16` — exact, no
    /// rounding, no table.
    Bf16(&'a [u16]),
    /// NVFP4 straight from a compiled pack: e2m1 codes two to the byte
    /// (lo nibble first), one E4M3 scale per sixteen elements along the
    /// input axis, and the single f32 tensor scale both levels are
    /// expressed relative to.
    ///
    /// Two scale levels rather than one because E4M3 cannot represent the
    /// product — that is the format's whole reason for existing, and it
    /// is why this cannot be folded into [`Self::Q8`]'s shape.
    ///
    /// Reaches the kernel compact, like every other variant here: the
    /// CPU-1B finding that widening to a scratch matrix costs more than
    /// the traffic it saves applies with more force to a 4-bit format,
    /// not less.
    Nvfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
        tensor_scale: f32,
    },
}

impl WeightRows<'_> {
    /// The address of the PRIMARY weight stream — this slab's identity.
    ///
    /// Two live operands cannot share a primary address, so this is what
    /// distinguishes one caller's projections from another's in a
    /// process-global record such as [`super::replay`]'s capture.
    pub fn primary_addr(&self) -> usize {
        match self {
            Self::F32(w) => w.as_ptr() as usize,
            Self::Bf16(w) => w.as_ptr() as usize,
            Self::Q8 { codes, .. } => codes.as_ptr() as usize,
            Self::Q4 { packed, .. } => packed.as_ptr() as usize,
            Self::Nvfp4 { packed, .. } => packed.as_ptr() as usize,
        }
    }

    /// Rows in this slab, given the column count.
    pub fn rows(&self, in_dim: usize) -> usize {
        match self {
            Self::F32(w) => w.len() / in_dim,
            Self::Bf16(w) => w.len() / in_dim,
            Self::Q8 { codes, .. } => codes.len() / in_dim,
            Self::Q4 { packed, .. } => packed.len() * 2 / in_dim,
            Self::Nvfp4 { packed, .. } => packed.len() * 2 / in_dim,
        }
    }

    /// A sub-slab of `count` rows starting at `start`.
    pub fn slice_rows(&self, in_dim: usize, start: usize, count: usize) -> Self {
        let (a, b) = (start * in_dim, (start + count) * in_dim);
        match self {
            Self::F32(w) => Self::F32(&w[a..b]),
            Self::Bf16(w) => Self::Bf16(&w[a..b]),
            // The scales are sliced ALONGSIDE the codes. A partition that
            // cut one and not the other would hand a worker the right
            // weights under another row's scales — finite, plausible
            // numbers, entirely wrong.
            Self::Q8 {
                codes,
                scales,
                sums,
                block,
            } => {
                let per_row = in_dim.div_ceil(*block);
                // The index is cut ALONGSIDE the codes, for the same
                // reason the scales are: a partition that moved one and
                // not the other would hand a worker another row's sums.
                let per_sum =
                    in_dim.div_ceil(crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK);
                Self::Q8 {
                    codes: &codes[a..b],
                    scales: &scales[start * per_row..(start + count) * per_row],
                    sums: if sums.is_empty() {
                        sums
                    } else {
                        &sums[start * per_sum..(start + count) * per_sum]
                    },
                    block: *block,
                }
            }
            Self::Q4 {
                packed,
                scales,
                block,
            } => {
                let per_row = in_dim.div_ceil(*block);
                Self::Q4 {
                    packed: &packed[a / 2..b / 2],
                    scales: &scales[start * per_row..(start + count) * per_row],
                    block: *block,
                }
            }
            Self::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => {
                // Groups run along the input axis, so a row slab is a
                // contiguous run of both streams. The tensor scale is
                // matrix-wide and travels with every slab unchanged.
                let groups = in_dim / larql_models::quant::nvfp4::NVFP4_GROUP_ELEMS;
                let per_row = groups * larql_models::quant::nvfp4::NVFP4_GROUP_BYTES;
                Self::Nvfp4 {
                    packed: &packed[start * per_row..(start + count) * per_row],
                    scales: &scales[start * groups..(start + count) * groups],
                    tensor_scale: *tensor_scale,
                }
            }
        }
    }

    /// Bytes this slab actually occupies — what the roofline is measured
    /// against, and the whole point of keeping it compact.
    pub fn bytes(&self) -> usize {
        match self {
            Self::F32(w) => w.len() * 4,
            Self::Bf16(w) => w.len() * 2,
            // Scales counted, because they are read too. A Q8 rate that
            // ignored its own metadata would flatter the format by the
            // exact amount the metadata costs.
            Self::Q8 {
                codes,
                scales,
                sums,
                ..
            } => codes.len() + scales.len() * 4 + sums.len() * 2,
            Self::Q4 { packed, scales, .. } => packed.len() + scales.len() * 4,
            // The tensor scale is one f32 for the whole matrix, not per
            // row, so it is not counted in a row slab's footprint.
            Self::Nvfp4 { packed, scales, .. } => packed.len() + scales.len(),
        }
    }
}
