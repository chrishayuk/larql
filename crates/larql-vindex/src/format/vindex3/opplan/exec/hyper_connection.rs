//! The hyper-connection residual topology, executed (wave 17).
//!
//! Wave 16 declared this topology and refused it. This module is the
//! refusal's answer: the five stages a hyper-connected sublayer runs,
//! each a function that can be called and judged on its own rather than
//! only as part of one opaque path. That separability is the whole
//! reason wave 16 named `hc_split_sinkhorn` as an operation instead of
//! burying it in the lowering.
//!
//! Read from DeepSeek-V4-Flash's own reference — `inference/model.py`
//! (`Block.hc_pre`, `Block.hc_post`, `ParallelHead.hc_head`,
//! `Transformer.forward`) and `inference/kernel.py`
//! (`hc_split_sinkhorn_kernel`). Wave 16 could transcribe the sublayer
//! equation but not the split, because the split lives in the kernel
//! file and its twenty iterations are exactly what a name does not tell
//! you.
//!
//! ```text
//! residual = x                                   // [streams, hidden]
//! mixes    = (hc_fn @ flatten(x)) * rsqrt(mean(flatten(x)^2) + norm_eps)
//! pre, post, comb = split_sinkhorn(mixes, hc_scale, hc_base, ..)
//! v   = Σ_j pre[j] * x[j]                        // streams -> ONE
//! b   = sublayer(norm(v))                        // the ordinary branch
//! out[k] = post[k] * b + Σ_j comb[j,k] * residual[j]   // ONE -> streams
//! ```
//!
//! **Three asymmetries in the reference are real and must not be tidied
//! away.** `pre` carries `+ eps` and no factor; `post` carries a factor
//! of two and no eps; and `hc_scale` is three different scalars indexed
//! by which output they scale. A transcription that regularised any of
//! them would compute a different model and still look reasonable.
//!
//! **The expansion's output stream is `comb`'s SECOND index** and the
//! sum runs over the first. Two independent facts fix that order: the
//! broadcast in `Block.hc_post`, and the observation that the split
//! ends on a COLUMN normalisation — so columns sum to one, and the
//! expansion is a weighted average over source streams only under this
//! reading.
//!
//! The head's reduction is a DIFFERENT operation, not a reuse of stages
//! one to three: [`head_reduce`] runs no split at all. See its own
//! documentation.

use super::kernels::{matvec, sigmoid, softmax};
use larql_models::config::HyperConnection;

/// Index of the scalar that scales the `pre` logits inside `hc_scale`.
pub const SCALE_PRE: usize = 0;
/// Index of the scalar that scales the `post` logits.
pub const SCALE_POST: usize = 1;
/// Index of the scalar that scales the combination logits.
pub const SCALE_COMB: usize = 2;
/// `hc_scale` carries exactly these three, and a checkpoint offering a
/// different count is describing a different operation.
pub const HC_SCALE_LEN: usize = 3;
/// The HEAD's `hc_head_scale` is a single scalar — one of the two shape
/// facts (with `[hc, hc·hidden]` against a site's `[(2 + hc)·hc, ..]`)
/// that make [`head_reduce`] a different operation from a site's split.
/// Named so the op plan checks the head's operand against the same
/// number the executor consumes.
pub const HC_HEAD_SCALE_LEN: usize = 1;

/// `post = 2 * sigmoid(..)`. The factor is the reference's, and it is
/// named because `pre` deliberately does NOT carry it — the two
/// half-equations differ, and a shared helper would erase that.
const POST_GAIN: f32 = 2.0;

/// The three outputs of one split: the reduction weights, the expansion
/// weights, and the matrix that mixes every stream into every other.
#[derive(Debug, Clone, PartialEq)]
pub struct SinkhornSplit {
    /// `[streams]` — how the bundle collapses to one vector.
    pub pre: Vec<f32>,
    /// `[streams]` — how the branch output is scattered back.
    pub post: Vec<f32>,
    /// `[streams * streams]`, row-major `[j, k]`: how much of source
    /// stream `j` reaches destination stream `k`.
    pub comb: Vec<f32>,
}

impl SinkhornSplit {
    /// `comb[j, k]` — source stream `j`, destination stream `k`.
    pub fn comb_at(&self, streams: usize, j: usize, k: usize) -> f32 {
        self.comb[j * streams + k]
    }
}

/// **Stage 1** — the dynamic mix projection.
///
/// `mixes = (hc_fn @ flatten(x)) * rsqrt(mean(flatten(x)^2) + norm_eps)`.
///
/// The epsilon here is the COMPONENT's `norm_eps`, not the topology's
/// `sinkhorn_eps`. The reference passes them separately and this build
/// must not merge them: they are 1e-6 in DeepSeek-V4-Flash and equal by
/// coincidence, not by definition.
///
/// The RMS statistic is taken over the FLATTENED bundle — one statistic
/// for all `streams * hidden` elements, not one per stream.
pub fn mix_projection(
    x: &[f32],
    streams: usize,
    hidden: usize,
    hc_fn: &[f32],
    mix_rows: usize,
    norm_eps: f64,
) -> Vec<f32> {
    let flat = streams * hidden;
    assert_eq!(x.len(), flat, "state is [streams, hidden]");
    assert_eq!(
        hc_fn.len(),
        mix_rows * flat,
        "hc_fn is [mix_rows, streams * hidden]"
    );

    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / flat as f32;
    let rsqrt = (1.0 / (mean_square as f64 + norm_eps).sqrt()) as f32;

    let mut mixes = matvec(hc_fn, mix_rows, flat, x);
    for m in mixes.iter_mut() {
        *m *= rsqrt;
    }
    mixes
}

/// **Stage 2** — `hc_split_sinkhorn`, transcribed from the kernel.
///
/// One `[mix_rows]` vector becomes `pre[streams]`, `post[streams]` and
/// `comb[streams, streams]`. `iterations` counts total normalisation
/// passes: the kernel spells the first row-softmax and column pass out
/// before its loop, so twenty iterations are one softmax pass, one
/// column pass, then nineteen row/column pairs.
///
/// This is iterative normalisation, not a reshape — running it once
/// instead of twenty times gives a measurably different matrix, and the
/// fixture asserts exactly that.
pub fn split_sinkhorn(
    mixes: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    hc: HyperConnection,
) -> SinkhornSplit {
    let streams = hc.streams;
    let mix_rows = mix_rows_for(streams);
    assert_eq!(mixes.len(), mix_rows, "mixes is [(2 + streams) * streams]");
    assert_eq!(
        hc_base.len(),
        mix_rows,
        "hc_base is [(2 + streams) * streams]"
    );
    assert_eq!(hc_scale.len(), HC_SCALE_LEN, "hc_scale is [3]");

    let eps = hc.sinkhorn_eps as f32;

    // pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps
    let pre = (0..streams)
        .map(|j| sigmoid(mixes[j] * hc_scale[SCALE_PRE] + hc_base[j]) + eps)
        .collect();

    // post[j] = 2 * sigmoid(mixes[j + streams] * scale[1] + base[..]) — no eps.
    let post = (0..streams)
        .map(|j| {
            let at = j + streams;
            POST_GAIN * sigmoid(mixes[at] * hc_scale[SCALE_POST] + hc_base[at])
        })
        .collect();

    // comb[j, k] reads flat index 2*streams + j*streams + k.
    let combination_base = 2 * streams;
    let mut comb = vec![0.0f32; streams * streams];
    for j in 0..streams {
        for k in 0..streams {
            let at = combination_base + j * streams + k;
            comb[j * streams + k] = mixes[at] * hc_scale[SCALE_COMB] + hc_base[at];
        }
    }

    sinkhorn_normalise(&mut comb, streams, hc.sinkhorn_iters, eps);
    SinkhornSplit { pre, post, comb }
}

/// The normalisation itself: a row softmax offset by `eps`, a column
/// pass, then `iterations - 1` row/column pairs.
///
/// Split out because it is the part with a convergence property worth
/// testing on its own — after it, every column sums to approximately
/// one, which is what makes the expansion a weighted average.
fn sinkhorn_normalise(comb: &mut [f32], streams: usize, iterations: usize, eps: f32) {
    // comb = comb.softmax(-1) + eps
    for j in 0..streams {
        softmax(&mut comb[j * streams..(j + 1) * streams]);
    }
    for c in comb.iter_mut() {
        *c += eps;
    }
    normalise_columns(comb, streams, eps);

    for _ in 1..iterations {
        normalise_rows(comb, streams, eps);
        normalise_columns(comb, streams, eps);
    }
}

/// `comb[j, k] /= Σ_k comb[j, k] + eps`.
fn normalise_rows(comb: &mut [f32], streams: usize, eps: f32) {
    for j in 0..streams {
        let row = &mut comb[j * streams..(j + 1) * streams];
        let sum: f32 = row.iter().sum::<f32>() + eps;
        for c in row.iter_mut() {
            *c /= sum;
        }
    }
}

/// `comb[j, k] /= Σ_j comb[j, k] + eps`. The split ENDS on this, which
/// is why columns sum to one and destination streams receive a weighted
/// average of the sources.
fn normalise_columns(comb: &mut [f32], streams: usize, eps: f32) {
    for k in 0..streams {
        let sum: f32 = (0..streams).map(|j| comb[j * streams + k]).sum::<f32>() + eps;
        for j in 0..streams {
            comb[j * streams + k] /= sum;
        }
    }
}

/// **Stage 3** — the bundle collapses to one vector.
///
/// `v[d] = Σ_j pre[j] * x[j, d]`.
pub fn reduce_streams(pre: &[f32], x: &[f32], streams: usize, hidden: usize) -> Vec<f32> {
    assert_eq!(pre.len(), streams, "one reduction weight per stream");
    assert_eq!(x.len(), streams * hidden, "state is [streams, hidden]");

    let mut reduced = vec![0.0f32; hidden];
    for (j, weight) in pre.iter().enumerate() {
        let stream = &x[j * hidden..(j + 1) * hidden];
        for (out, v) in reduced.iter_mut().zip(stream) {
            *out += weight * v;
        }
    }
    reduced
}

/// **Stage 5** — the branch output is scattered back across the bundle.
///
/// `out[k, d] = post[k] * branch[d] + Σ_j comb[j, k] * residual[j, d]`.
///
/// Note which index is which: `k` is the DESTINATION stream and the sum
/// runs over sources `j`. Transposing `comb` here produces a different
/// and entirely plausible-looking bundle, so the fixture asserts the
/// transposed form disagrees.
pub fn expand_streams(
    branch: &[f32],
    residual: &[f32],
    split: &SinkhornSplit,
    streams: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(branch.len(), hidden, "the branch returns one vector");
    assert_eq!(
        residual.len(),
        streams * hidden,
        "residual is [streams, hidden]"
    );
    assert_eq!(split.post.len(), streams, "one expansion weight per stream");
    assert_eq!(
        split.comb.len(),
        streams * streams,
        "comb is [streams, streams]"
    );

    let mut out = vec![0.0f32; streams * hidden];
    for k in 0..streams {
        let destination = &mut out[k * hidden..(k + 1) * hidden];
        let post_k = split.post[k];
        for (o, b) in destination.iter_mut().zip(branch) {
            *o = post_k * b;
        }
        for j in 0..streams {
            let weight = split.comb_at(streams, j, k);
            let source = &residual[j * hidden..(j + 1) * hidden];
            for (o, r) in destination.iter_mut().zip(source) {
                *o += weight * r;
            }
        }
    }
    out
}

/// The embedding entering the stack: one vector REPLICATED into every
/// stream.
///
/// `Transformer.forward` does `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`
/// — not a zero pad, not a split, and not a learned expansion. Named
/// here because wave 16 identified the embedding as a consumer that has
/// to agree with the topology without recording what it actually does.
pub fn expand_embedding(h: &[f32], streams: usize) -> Vec<f32> {
    let mut bundle = Vec::with_capacity(streams * h.len());
    for _ in 0..streams {
        bundle.extend_from_slice(h);
    }
    bundle
}

/// The head's OWN reduction — a sixth site, and a DIFFERENT operation
/// from stages one to three.
///
/// `ParallelHead.hc_head` runs no split: no iterations, no `post`, no
/// combination matrix. Its `hc_head_fn` is `[streams, streams * hidden]`
/// rather than `[mix_rows, ..]`, and its `hc_head_scale` is a single
/// scalar rather than three. It is
/// `pre = sigmoid(mixes * scale + base) + hc_eps` followed by the same
/// weighted sum stage three performs.
///
/// Wave 16 recorded that the head "carries its own reduction", which is
/// true and understates it; treating it as a reuse of the sublayer path
/// would apply a Sinkhorn the reference does not run.
pub fn head_reduce(
    x: &[f32],
    streams: usize,
    hidden: usize,
    head: &HeadWeights<'_>,
    norm_eps: f64,
    hc_eps: f64,
) -> Vec<f32> {
    assert_eq!(head.base.len(), streams, "the head's base is [streams]");
    // ONE row per stream, where a sublayer site has (2 + streams) *
    // streams — the shape difference is the operation difference.
    let mixes = mix_projection(x, streams, hidden, head.reduce_fn, streams, norm_eps);
    let pre: Vec<f32> = mixes
        .iter()
        .zip(head.base)
        .map(|(m, b)| sigmoid(m * head.scale + b) + hc_eps as f32)
        .collect();
    reduce_streams(&pre, x, streams, hidden)
}

/// `(2 + streams) * streams` — the mix projection's row count, derived
/// so it cannot drift from the stream count.
pub fn mix_rows_for(streams: usize) -> usize {
    (2 + streams) * streams
}

/// The head's own operands — a different shape from a sublayer site's,
/// because [`head_reduce`] is a different operation.
///
/// `reduce_fn` has ONE row per stream where [`SiteWeights::mix_fn`] has
/// `(2 + streams) * streams`, and `scale` is a single scalar where a
/// site carries three. Separate types rather than one with optional
/// fields: a head that accidentally received a site's operands would
/// otherwise be a runtime length check instead of a compile error.
pub struct HeadWeights<'a> {
    /// `[streams, streams * hidden]`.
    pub reduce_fn: &'a [f32],
    /// `[streams]`.
    pub base: &'a [f32],
    /// A single scalar, not the site's three.
    pub scale: f32,
}

/// The operands one hyper-connection site owns. Attention and FFN each
/// have their own set, and the head has a different shape entirely.
pub struct SiteWeights<'a> {
    /// `[mix_rows, streams * hidden]`.
    pub mix_fn: &'a [f32],
    /// `[mix_rows]`.
    pub base: &'a [f32],
    /// `[3]`, indexed by [`SCALE_PRE`], [`SCALE_POST`], [`SCALE_COMB`].
    pub scale: &'a [f32],
}

/// All five stages, composed — one hyper-connected sublayer.
///
/// `branch` is stage four: it receives the REDUCED `[hidden]` vector and
/// returns one, exactly as an ordinary sublayer would. Everything that
/// makes this topology different from `h + f(h)` lives on either side of
/// it, which is why the ordinary operators need no changes at all.
///
/// The single-position reference composition: [`reduce`], the branch,
/// [`update`]. The decode traversal runs the same two halves around its
/// own operator dispatch rather than calling this, because the branch
/// there borrows the session's continuation state — but the halves ARE
/// these, so the two cannot compute different sublayers.
pub fn sublayer_forward(
    x: &[f32],
    streams: usize,
    hidden: usize,
    site: &SiteWeights<'_>,
    hc: HyperConnection,
    norm_eps: f64,
    branch: impl FnOnce(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    debug_assert_eq!(streams, hc.streams, "the site and the topology must agree");
    let bundle = Bundle::from_flat(streams, hidden, x.to_vec()).unwrap_or_else(|e| panic!("{e}"));
    let entry = reduce(&bundle, site, hc, norm_eps, Mutation::None);
    let branched = branch(&entry.reduced);
    update(&bundle, &branched, &entry.split, Mutation::None).into_flat()
}

/// The residual carrier of a hyper-connected component at ONE position:
/// `streams` parallel `[hidden]` vectors, held row-major `[streams,
/// hidden]` exactly as the five stages read it.
///
/// A type, not a wider `Vec<f32>`: the stream count and the hidden width
/// are different semantic dimensions. Every ordinary operator keeps
/// consuming `[hidden]`, and nothing outside this module can hand a
/// bundle to one by mistake — the flat view is explicit, and a
/// `[hidden]` vector comes out of it only through [`reduce`] (a site)
/// or [`head_reduce`] (the stack's end).
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    streams: usize,
    hidden: usize,
    data: Vec<f32>,
}

impl Bundle {
    /// The embedding entering the stack, replicated into every stream —
    /// [`expand_embedding`] as a typed value.
    pub fn replicate(h: &[f32], streams: usize) -> Self {
        Self {
            streams,
            hidden: h.len(),
            data: expand_embedding(h, streams),
        }
    }

    /// A bundle from its flat `[streams * hidden]` form, refused when the
    /// length does not factor.
    pub fn from_flat(streams: usize, hidden: usize, data: Vec<f32>) -> Result<Self, String> {
        if streams == 0 || hidden == 0 || data.len() != streams * hidden {
            return Err(format!(
                "a bundle of {streams} streams x {hidden} hidden needs {} values, got {}",
                streams * hidden,
                data.len()
            ));
        }
        Ok(Self {
            streams,
            hidden,
            data,
        })
    }

    /// A bundle from one `[hidden]` row per stream, refused when the rows
    /// disagree in width or there are none.
    pub fn from_streams(rows: &[Vec<f32>]) -> Result<Self, String> {
        let Some(first) = rows.first() else {
            return Err("a bundle needs at least one stream".to_string());
        };
        let hidden = first.len();
        if rows.iter().any(|r| r.len() != hidden) {
            return Err("every stream of a bundle has the same hidden width".to_string());
        }
        Self::from_flat(rows.len(), hidden, rows.concat())
    }

    pub fn streams(&self) -> usize {
        self.streams
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Stream `j`'s `[hidden]` vector.
    pub fn stream(&self, j: usize) -> &[f32] {
        &self.data[j * self.hidden..(j + 1) * self.hidden]
    }

    pub fn stream_mut(&mut self, j: usize) -> &mut [f32] {
        &mut self.data[j * self.hidden..(j + 1) * self.hidden]
    }

    /// The `[streams, hidden]` view the stages read. Explicitly named so
    /// a reader can see where a bundle is handed to topology arithmetic.
    pub fn as_flat(&self) -> &[f32] {
        &self.data
    }

    pub fn into_flat(self) -> Vec<f32> {
        self.data
    }

    /// The elementwise mean over streams — NOT a reduction the topology
    /// defines. It exists for the negative controls, which need a
    /// plausible wrong `[hidden]` vector to feed where the reduced one
    /// belongs.
    pub fn stream_mean(&self) -> Vec<f32> {
        let mut mean = vec![0.0f32; self.hidden];
        for j in 0..self.streams {
            for (m, v) in mean.iter_mut().zip(self.stream(j)) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= self.streams as f32;
        }
        mean
    }
}

// The control vocabulary moved to `super::controls` when a SECOND
// residual topology needed its own defects: one value threads through
// one traversal, so the enum is one type, but it stopped belonging to
// hyper-connections the moment it stopped being only about them.
// Re-exported here so every `hyper_connection::Mutation` path — and
// there are hundreds, nearly all `Mutation::None` in test call sites —
// keeps resolving.
pub use super::controls::Mutation;

/// What one site's reduction produced: the split (stages one and two,
/// kept because stage five needs it and because it is the state a
/// witness can observe) and the reduced vector the branch sees.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteReduction {
    pub split: SinkhornSplit,
    pub reduced: Vec<f32>,
}

/// Stages one to three at one site: the bundle collapses to the ONE
/// `[hidden]` vector the ordinary operator consumes.
pub fn reduce(
    x: &Bundle,
    site: &SiteWeights<'_>,
    hc: HyperConnection,
    norm_eps: f64,
    mutation: Mutation,
) -> SiteReduction {
    let (streams, hidden) = (x.streams(), x.hidden());
    debug_assert_eq!(
        streams, hc.streams,
        "the bundle and the topology must agree"
    );
    let mixes = mix_projection(
        x.as_flat(),
        streams,
        hidden,
        site.mix_fn,
        mix_rows_for(streams),
        norm_eps,
    );
    let hc_run = match mutation {
        Mutation::SingleIteration => HyperConnection {
            sinkhorn_iters: 1,
            ..hc
        },
        _ => hc,
    };
    let mut split = split_sinkhorn(&mixes, site.scale, site.base, hc_run);
    if mutation == Mutation::UniformReduction {
        split.pre = vec![1.0 / streams as f32; streams];
    }
    let reduced = reduce_streams(&split.pre, x.as_flat(), streams, hidden);
    SiteReduction { split, reduced }
}

/// Stage five at one site: the branch's output is scattered back and the
/// bundle carried forward through `comb`. This REPLACES the residual
/// add — the carry and the add are one operation here.
pub fn update(x: &Bundle, branch: &[f32], split: &SinkhornSplit, mutation: Mutation) -> Bundle {
    let (streams, hidden) = (x.streams(), x.hidden());
    let transposed;
    let split = if mutation == Mutation::TransposedCombination {
        let mut comb = vec![0.0f32; streams * streams];
        for j in 0..streams {
            for k in 0..streams {
                comb[j * streams + k] = split.comb_at(streams, k, j);
            }
        }
        transposed = SinkhornSplit {
            pre: split.pre.clone(),
            post: split.post.clone(),
            comb,
        };
        &transposed
    } else {
        split
    };
    Bundle {
        streams,
        hidden,
        data: expand_streams(branch, x.as_flat(), split, streams, hidden),
    }
}
