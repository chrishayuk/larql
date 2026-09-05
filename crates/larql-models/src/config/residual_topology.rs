//! How a component's residual stream is shaped and recombined.
//!
//! Every family judged before 2026 carries ONE residual vector and adds
//! each sublayer's output into it. That is [`ResidualTopology::SingleStream`],
//! and it is stated rather than assumed so that a different topology is a
//! different VALUE here and not an unnoticed absence.
//!
//! The second judged topology is hyper-connections, read from
//! DeepSeek-V4-Flash's own `inference/model.py` (`class Block`):
//!
//! ```text
//! residual = x                                  // [b, s, hc, d] — hc STREAMS
//! mixes    = Linear(x.flatten(2), hc_fn) * rsqrt(mean(x^2) + eps)
//! pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)
//! x = sum(pre * x, dim=2)                       // hc streams -> ONE vector
//! x = sublayer(sublayer_norm(x))                // the ordinary branch
//! x = post * x + sum(comb * residual, dim=2)    // ONE -> hc streams
//! ```
//!
//! **This is not a scale on an ordinary residual**, and three independent
//! facts each rule that reading out:
//!
//! 1. the state SHAPE differs — `hc` streams, not one;
//! 2. the reduce and expand weights are DYNAMIC, computed per token from
//!    the current state through `hc_*_fn`, not stored per layer;
//! 3. the expand mixes every stream into every other through
//!    `comb[hc, hc]`, so the streams are not independent residuals
//!    running in parallel.
//!
//! Any one of them means a `SingleStream` programme cannot lower this
//! without discarding something the checkpoint declares.
//!
//! **The stream count is a COMPONENT fact, not a layer fact.** Once the
//! residual means `[..., hc, d]`, every consumer has to know: the
//! embedding produces one vector where the stack expects a bundle, the
//! branch operators receive the reduced `[..., d]`, and the head carries
//! its own reduction (`hc_head_{fn,base,scale}`). A per-layer flag would
//! let a stack claim hyper-connections while its embedding and head
//! silently assumed one stream.

use serde::{Deserialize, Serialize};

/// How the residual stream is shaped and recombined across a component.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResidualTopology {
    /// One residual vector; each sublayer's output is added into it.
    /// Every family judged before hyper-connections.
    SingleStream,
    /// A bundle of parallel residual streams, reduced to one vector for
    /// each sublayer and expanded back afterwards, with per-token weights.
    HyperConnection(HyperConnection),
}

/// The declared parameters of a hyper-connection topology. Every field is
/// read from the checkpoint's config; none has a default, because a
/// wrong stream count or iteration count computes a different model
/// rather than failing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HyperConnection {
    /// `hc_mult` — how many parallel residual streams the state carries.
    pub streams: usize,
    /// `hc_sinkhorn_iters` — iterations of the normalisation that splits
    /// the projected statistics into `pre`, `post` and `comb`.
    pub sinkhorn_iters: usize,
    /// `hc_eps` — the epsilon that split runs at. Distinct from the
    /// component's `norm_eps`, which the mix projection's RMS uses; the
    /// reference passes them separately and this build must not merge
    /// them.
    pub sinkhorn_eps: f64,
}

impl ResidualTopology {
    /// The number of parallel residual streams the state carries. One for
    /// every topology but hyper-connections.
    pub fn streams(self) -> usize {
        match self {
            Self::SingleStream => 1,
            Self::HyperConnection(hc) => hc.streams,
        }
    }

    /// Whether this is the one-vector residual every family before
    /// hyper-connections carries. Serde reads it to leave a single-stream
    /// plan's serialisation byte-identical to what it was before the
    /// topology travelled on the plan at all.
    pub fn is_single_stream(&self) -> bool {
        matches!(self, Self::SingleStream)
    }
}

/// One operation in the execution language, named separately rather than
/// buried inside hyper-connection lowering.
///
/// `hc_split_sinkhorn` runs `sinkhorn_iters` iterative normalisation
/// steps — it is neither metadata preprocessing nor a reshape. Naming it
/// here is what lets its five stages be judged independently: the mix
/// projection, this split, the stream reduction, the branch, and the
/// stream expansion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HyperConnectionWeights {
    /// Rows of the `[mix_hc, streams * hidden]` projection that reads the
    /// current state — `mix_hc = (2 + streams) * streams`.
    pub mix_rows: usize,
    pub streams: usize,
    pub iterations: usize,
    pub epsilon: f64,
}

impl HyperConnectionWeights {
    /// `(2 + hc) * hc` — the projection's row count, derived so it cannot
    /// drift from the stream count.
    pub fn mix_rows_for(streams: usize) -> usize {
        (2 + streams) * streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both judged topologies lower: the refusal that lived here through
    /// waves 16-18 was retired in wave 19, after the decode and batch
    /// traversals were each witnessed against the reference's oracle.
    /// What the type still says is the stream count and which of the two
    /// residual programmes a component runs.
    #[test]
    fn both_judged_topologies_lower() {
        assert_eq!(ResidualTopology::SingleStream.streams(), 1);
        assert!(ResidualTopology::SingleStream.is_single_stream());

        let hc = ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        });
        assert_eq!(hc.streams(), 4);
        assert!(!hc.is_single_stream());
    }

    /// The mix projection's row count is derived from the stream count,
    /// never declared beside it: `(2 + hc) * hc`, which for the four
    /// streams both real checkpoints declare is the 24 rows the wave-17
    /// oracle and every site operand carry.
    #[test]
    fn mix_rows_derive_from_the_stream_count() {
        assert_eq!(HyperConnectionWeights::mix_rows_for(4), 24);
        assert_eq!(HyperConnectionWeights::mix_rows_for(1), 3);
        assert_eq!(HyperConnectionWeights::mix_rows_for(8), 80);
    }
}
