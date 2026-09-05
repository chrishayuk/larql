//! The attention-residual topology's state and arithmetic
//! (K3-ATTNRES-1, transition 2a).
//!
//! Transcribed from Kimi-K3's own `modeling_kimi_linear.py` by way of
//! the rung's ORACLE (`scripts/attn_res_oracle_export.py`, committed
//! `ec7da08d`), which read that file verbatim and exported a per-site
//! witness plus one rejecting control per named ordering mistake. This
//! module is checked against that export; it is not the reference, and
//! it must never become one.
//!
//! # The state is a HISTORY, not a bundle
//!
//! ```text
//! History { prefix: [hidden], snapshots: [N, hidden] }
//! ```
//!
//! and the distinction from [`super::hyper_connection::Bundle`] is
//! semantic rather than a matter of width. A bundle's streams are
//! interchangeable parallel residuals whose count is DECLARED once and
//! fixed for the whole stack. Snapshots are ordered historical states of
//! the prefix, produced by block-boundary EVENTS, whose count is a
//! function of depth and grows as the stack runs. Reaching for `Bundle`
//! because it also holds more than one vector is how a second topology
//! becomes a wrong dialect of the first.
//!
//! # The prefix is genuinely absent for part of a boundary layer
//!
//! The reference sets `prefix_sum = None` at a block boundary and lets
//! the attention branch's output BECOME the new prefix rather than being
//! added to one. That window — between the boundary event and the
//! attention branch — is the only time a prefix does not exist, and
//! nothing reduces during it. [`History::prefix`] is therefore an
//! `Option`, and a reduction over an absent prefix is an executor bug
//! rather than a model to run anyway.

use crate::error::VindexError;

use super::controls::Mutation;

/// One token's residual state under the attention-residual topology.
///
/// Cloned per position by the batch traversal, carried per step by the
/// decode one. Both hold the same thing, which is the point of the type.
#[derive(Debug, Clone, PartialEq)]
pub struct History {
    hidden: usize,
    /// `None` only between a boundary event and the attention branch
    /// that supplies the new prefix. See the module docs.
    prefix: Option<Vec<f32>>,
    /// Block-boundary snapshots of the ENTERING prefix state, oldest
    /// first. Order is load-bearing: the exit and every site read them
    /// as a sequence, and the oracle's probabilities are per candidate.
    snapshots: Vec<Vec<f32>>,
}

impl History {
    /// The state entering the stack: the embedded vector, and no
    /// snapshots at all. The reference starts `block_residual` at
    /// `new_zeros(tokens, 0, hidden)` — an EMPTY set, which is what
    /// makes layer 0's attention site have nothing to read.
    pub fn new(prefix: Vec<f32>) -> Self {
        Self {
            hidden: prefix.len(),
            prefix: Some(prefix),
            snapshots: Vec::new(),
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// How many snapshots the history holds. The witness records this
    /// BEFORE each site, because the whole ordering claim of the
    /// topology is which set a site read.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn snapshots(&self) -> &[Vec<f32>] {
        &self.snapshots
    }

    /// The prefix, or `None` inside the boundary window.
    pub fn prefix(&self) -> Option<&[f32]> {
        self.prefix.as_deref()
    }

    /// How many candidates a reduction over this state would mix:
    /// every snapshot plus the prefix. Never fewer than two anywhere in
    /// the reference's schedule — the oracle measured that, and the one
    /// site that would see a single candidate is layer 0's attention
    /// site, where the reference does not reduce at all.
    pub fn candidate_count(&self) -> usize {
        self.snapshots.len() + usize::from(self.prefix.is_some())
    }

    /// The boundary event's first half: append a snapshot. The value is
    /// the caller's, because WHICH vector is snapshotted is exactly what
    /// two of the rung's controls perturb.
    pub fn push_snapshot(&mut self, value: Vec<f32>) {
        self.snapshots.push(value);
    }

    /// The boundary event's second half: the prefix ceases to exist
    /// until a branch supplies one.
    pub fn reset_prefix(&mut self) {
        self.prefix = None;
    }

    /// A site's write: add the branch's delta into the prefix, or — when
    /// a boundary reset it — BECOME the prefix. One method because the
    /// reference is one expression (`prefix_sum + hidden_states` or
    /// `hidden_states`), and splitting it would let a caller forget the
    /// second arm.
    pub fn write(&mut self, delta: &[f32]) {
        match &mut self.prefix {
            Some(prefix) => {
                for (p, d) in prefix.iter_mut().zip(delta) {
                    *p += *d;
                }
            }
            None => self.prefix = Some(delta.to_vec()),
        }
    }

    /// The prefix, consumed — the stack's output before the exit
    /// reduction. Refuses inside the boundary window, which no caller
    /// can be in.
    pub fn into_prefix(self) -> Result<Vec<f32>, VindexError> {
        self.prefix.ok_or_else(|| {
            VindexError::Parse(
                "the attention-residual history has no prefix: a boundary reset it and no \
                 branch supplied one. This is an executor bug — the schedule always writes \
                 before it reads again"
                    .to_string(),
            )
        })
    }
}

/// One site's two operands. A `[hidden]` norm weight and a
/// `[1, hidden]` projection, multiplied elementwise into ONE learned
/// score vector — there is no query and no per-token projection of the
/// state, which is why the pair is two stored vectors rather than a mix
/// projection.
#[derive(Debug, Clone, Copy)]
pub struct SitePair<'a> {
    pub norm: &'a [f32],
    pub proj: &'a [f32],
}

/// What one reduction produced: the distribution over candidates, and
/// the mixed vector the branch consumes.
///
/// `probs` is kept because it is the state a witness can observe and the
/// reference does not return — a single-stream traversal has no such
/// object at all, so it is the value that says the topology ran.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    pub probs: Vec<f32>,
    pub mixed: Vec<f32>,
}

/// `_apply_attn_res`, transcribed.
///
/// ```text
/// v      = cat(snapshots, prefix)         [N + 1, hidden]
/// k      = v * rsqrt(mean(v^2) + eps)     per candidate, no weight
/// score  = sum_h k_h * (norm_h * proj_h)  ONE learned vector
/// probs  = softmax(score)
/// out    = probs @ v                      the RAW candidates
/// ```
///
/// The two places a transcription can go wrong are which tensor is
/// SCORED and which is MIXED, and both are controls rather than
/// comments: the oracle measures 5.83e-01 for scoring the raw
/// candidates and 1.30e+00 for mixing the normalised ones.
pub fn reduce(
    history: &History,
    pair: SitePair<'_>,
    eps: f64,
    mutation: Mutation,
) -> Result<Reduction, VindexError> {
    let Some(prefix) = history.prefix() else {
        return Err(VindexError::Parse(
            "an attention-residual site reduced over a history whose prefix a boundary had \
             reset; nothing in the reference's schedule reads during that window"
                .to_string(),
        ));
    };
    let hidden = history.hidden();
    if pair.norm.len() != hidden || pair.proj.len() != hidden {
        return Err(VindexError::Parse(format!(
            "the site pair is [{}] and [{}] against a [{hidden}] residual",
            pair.norm.len(),
            pair.proj.len()
        )));
    }
    // The candidates, in the reference's order: every snapshot oldest
    // first, then the prefix last.
    let candidates: Vec<&[f32]> = history
        .snapshots()
        .iter()
        .map(|s| s.as_slice())
        .chain(std::iter::once(prefix))
        .collect();

    // The score vector's two factors, multiplied once.
    let score_weight: Vec<f32> = pair
        .norm
        .iter()
        .zip(pair.proj)
        .map(|(n, p)| n * p)
        .collect();

    let eps = eps as f32;
    let mix_normalised = mutation == Mutation::AttnResMixOverNormalisedCandidates;
    let score_raw = mutation == Mutation::AttnResScoreWithoutRmsNorm;

    let mut normalised: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());
    let mut scores: Vec<f32> = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let variance = candidate.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let scale = (variance + eps).sqrt().recip();
        let k: Vec<f32> = candidate.iter().map(|v| v * scale).collect();
        let scored: &[f32] = if score_raw { candidate } else { &k };
        scores.push(scored.iter().zip(&score_weight).map(|(v, w)| v * w).sum());
        normalised.push(k);
    }

    // Softmax, stabilised the way torch stabilises it, so the comparison
    // against the oracle is not a comparison of two overflow policies.
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= total;
    }

    let mut mixed = vec![0.0f32; hidden];
    for (index, weight) in probs.iter().enumerate() {
        let source: &[f32] = if mix_normalised {
            &normalised[index]
        } else {
            candidates[index]
        };
        for (m, v) in mixed.iter_mut().zip(source) {
            *m += weight * v;
        }
    }
    Ok(Reduction { probs, mixed })
}

/// Whether `layer` starts a block, and therefore carries the boundary
/// event. `layer_idx % block_size == 0` — the reference's own
/// expression, which makes layer 0 a boundary and is why the snapshot
/// set is never empty after it.
pub fn is_block_boundary(layer: usize, block_size: usize) -> bool {
    // The zero guard is kept in front deliberately, and is not what the
    // lint asked about: `0usize.is_multiple_of(0)` is TRUE, so a period
    // of zero would make layer 0 a boundary. No checkpoint declares one
    // — the topology is read as declared or not at all — and this
    // answers "no boundaries" rather than "one at the top" if some
    // future estate ever does.
    block_size != 0 && layer.is_multiple_of(block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule's one declared fact, and the guard in front of it.
    ///
    /// The zero arm is not defensive noise: clippy's suggested rewrite of
    /// this function was `layer.is_multiple_of(block_size)` ALONE, and
    /// that changes the answer for a period of zero — `is_multiple_of`
    /// returns true when both operands are zero, so layer 0 would become
    /// a boundary. This asserts the difference rather than quoting the
    /// standard library's contract at it, which is the only way the
    /// comment on the guard can be trusted.
    #[test]
    fn boundaries_fall_where_the_period_divides_and_a_zero_period_has_none() {
        // K3 declares 12 over 93 layers; the oracle's fixture 3 over 7.
        for (layer, expected) in [
            (0, true),
            (1, false),
            (2, false),
            (3, true),
            (4, false),
            (5, false),
            (6, true),
            (7, false),
        ] {
            assert_eq!(is_block_boundary(layer, 3), expected, "layer {layer}");
        }
        assert!(is_block_boundary(0, 12));
        assert!(is_block_boundary(84, 12));
        assert!(!is_block_boundary(11, 12));

        // The guard, and the std behaviour it exists to override.
        assert!(0usize.is_multiple_of(0), "the premise of the guard");
        assert!(
            !is_block_boundary(0, 0),
            "a zero period declares no schedule"
        );
        assert!(!is_block_boundary(5, 0));
    }

    /// The history's two-arm write, which the reference spells as one
    /// expression: add into the prefix, or BECOME it after a boundary
    /// reset. A caller that only implemented the first arm would produce
    /// a plausible vector at every layer except the boundaries.
    #[test]
    fn a_write_after_a_boundary_reset_becomes_the_prefix() {
        let mut history = History::new(vec![1.0, 2.0]);
        assert_eq!(history.candidate_count(), 1);
        history.write(&[0.5, 0.5]);
        assert_eq!(history.prefix(), Some([1.5, 2.5].as_slice()));

        history.push_snapshot(vec![9.0, 9.0]);
        assert_eq!(history.snapshot_count(), 1);
        assert_eq!(history.candidate_count(), 2);

        history.reset_prefix();
        assert_eq!(history.prefix(), None);
        // The snapshot survives the reset — the event mutates the
        // history and the prefix independently.
        assert_eq!(history.candidate_count(), 1);
        history.write(&[3.0, 4.0]);
        assert_eq!(history.prefix(), Some([3.0, 4.0].as_slice()));
        assert_eq!(history.candidate_count(), 2);
    }

    /// Reducing inside the boundary window is an executor bug, and says
    /// so rather than picking a vector to stand in for the absent prefix.
    #[test]
    fn a_reduction_over_an_absent_prefix_refuses() {
        let mut history = History::new(vec![1.0, 2.0]);
        history.push_snapshot(vec![3.0, 4.0]);
        history.reset_prefix();
        let norm = [1.0, 1.0];
        let proj = [0.1, 0.1];
        let err = reduce(
            &history,
            SitePair {
                norm: &norm,
                proj: &proj,
            },
            1e-5,
            Mutation::None,
        )
        .expect_err("no site reads during the boundary window");
        assert!(err.to_string().contains("boundary"), "{err}");
    }
}
