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

    /// Why this build cannot EXECUTE this topology, when it cannot.
    ///
    /// ONE authority, two readers: the executor's preparation step
    /// (`larql-vindex`'s `opplan::exec::prepared`), which refuses before a
    /// single operand is loaded, and the plan report, which must say so.
    /// Wave 11 established that a refusal only one consumer can see is
    /// not a refusal, and that two lists of what cannot be lowered drift
    /// into exactly that state.
    ///
    /// **The op plan is deliberately NOT a reader.** Wave 18 made the
    /// topology a closure question there — the six per-layer site
    /// operands classify, are required, are checked against the
    /// topology's own geometry and are bound into the plan — and refusing
    /// at that stage would hide the addressability answer behind the
    /// traversal gap, which is the structural silence wave 18's baseline
    /// recorded.
    ///
    /// **Three waves have each changed what this reason says without
    /// changing whether it refuses.** Wave 16 named the three structural
    /// facts a single-stream programme would have to discard. Wave 17
    /// implemented the arithmetic, so the reason stopped claiming it was
    /// missing. Wave 18 bound the operands, so it stops claiming the
    /// `hc_*` groups have no owner. What remains is the traversal: the
    /// generic executor carries one residual vector through the stack,
    /// and no real-weight witness has run the bundle from a checkpoint's
    /// own operands.
    ///
    /// Saying so precisely matters more than it looks. A stale reason
    /// sends the next wave to build something that exists; a build that
    /// lifted the refusal because the operands are addressable would be
    /// claiming execution from addressing — the same mistake as grading
    /// a config key representable because a parser read it.
    pub fn unimplemented_reason(self) -> Option<&'static str> {
        match self {
            Self::SingleStream => None,
            Self::HyperConnection(_) => Some(
                "the residual is a bundle of parallel streams reduced and expanded per token \
                 through a Sinkhorn-split mixing matrix; the single-stream residual programme \
                 cannot lower it without discarding the stream multiplicity, the dynamic \
                 reduce and expand weights, or the cross-stream combination. The five stages \
                 are implemented and checked against the reference, so the arithmetic is not \
                 what is missing; the hc_attn_*, hc_ffn_* and hc_head_* operands are addressed \
                 by role and bound into the op plan, so addressing is not what is missing \
                 either. What is missing is the traversal: the generic executor carries ONE \
                 residual vector through the stack, and no real-weight witness has run the \
                 bundle, the head reduction or the embedding replication from a checkpoint's \
                 own operands",
            ),
        }
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

    #[test]
    fn single_stream_lowers_and_hyper_connections_refuse() {
        assert_eq!(ResidualTopology::SingleStream.streams(), 1);
        assert!(ResidualTopology::SingleStream
            .unimplemented_reason()
            .is_none());

        let hc = ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        });
        assert_eq!(hc.streams(), 4);
        let reason = hc.unimplemented_reason().expect("must refuse");
        assert!(reason.contains("bundle"), "{reason}");
    }

    /// Wave 17 implemented the arithmetic and wave 18 bound the operands,
    /// so the refusal must claim neither is missing — and must name the
    /// gap that actually blocks execution, which is the traversal.
    ///
    /// A refusal whose stated reason has gone stale is worse than a
    /// blunt one: it sends the next wave to fix something already done.
    /// Wave 17's reason said "no placement rule owns the hc_* groups";
    /// that sentence became false the moment the roles landed, and this
    /// test is what makes it fail to compile as a claim.
    #[test]
    fn the_refusal_names_the_traversal_gap_and_neither_the_arithmetic_nor_the_addressing() {
        let hc = ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        });
        let reason = hc.unimplemented_reason().expect("still refuses");

        assert!(
            reason.contains("implemented"),
            "the reason must record that the arithmetic exists: {reason}"
        );
        assert!(
            reason.contains("the arithmetic is not what is missing"),
            "the reason must not leave a reader thinking the stages are unwritten: {reason}"
        );
        assert!(
            reason.contains("addressing is not what is missing"),
            "the reason must record that the operands are bound: {reason}"
        );
        assert!(
            !reason.contains("no placement rule"),
            "wave 17's addressability claim is stale after wave 18: {reason}"
        );
        assert!(
            reason.contains("traversal"),
            "the reason must name what actually blocks execution: {reason}"
        );
        // Wave 16's three structural reasons survive verbatim: they say
        // why a SINGLE-STREAM programme cannot lower this, which is still
        // true and is a different claim from what blocks execution today.
        for structural in ["stream multiplicity", "dynamic", "cross-stream"] {
            assert!(
                reason.contains(structural),
                "reason omits {structural:?}: {reason}"
            );
        }
    }

    #[test]
    fn only_the_single_stream_topology_is_single_stream() {
        assert!(ResidualTopology::SingleStream.is_single_stream());
        assert!(!ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        })
        .is_single_stream());
    }

    /// DeepSeek-V4's own numbers: `hc_mult` 4 gives a `[24, 4*d]`
    /// projection, and the row count is derived rather than stored.
    #[test]
    fn the_mix_projection_row_count_is_derived_from_the_stream_count() {
        assert_eq!(HyperConnectionWeights::mix_rows_for(4), 24);
        assert_eq!(HyperConnectionWeights::mix_rows_for(1), 3);
    }

    #[test]
    fn the_topology_round_trips_as_a_tagged_enum() {
        let hc = ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        });
        let json = serde_json::to_string(&hc).expect("serialise");
        assert!(json.contains("hyper_connection"), "{json}");
        assert_eq!(
            serde_json::from_str::<ResidualTopology>(&json).expect("round trip"),
            hc
        );
    }
}
