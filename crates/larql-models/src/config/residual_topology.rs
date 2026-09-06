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
//!
//! The third judged topology is **attention residuals**, read from
//! Kimi-K3's own `modeling_kimi_linear.py` (`KimiDecoderLayer.
//! _forward_attn_residual`, `_apply_attn_res`, `KimiLinearModel.forward`):
//!
//! ```text
//! prefix_sum = h_in;  blocks = []            // state: ONE vector + a history
//! per layer L:
//!   if blocks:  h = apply(prefix_sum, blocks, self_attention_res_*)
//!   if L % B == 0:  blocks.push(h_in); prefix_sum = None
//!   a = attention(input_layernorm(h));  prefix_sum = prefix_sum + a (or a)
//!   h = apply(prefix_sum, blocks, mlp_res_*)              // ALWAYS
//!   m = ffn(post_attention_layernorm(h)); prefix_sum += m
//! exit:  h = apply(prefix_sum, blocks, output_attn_res_*)  // REQUIRED
//!
//! apply(prefix, blocks, proj, norm):
//!   v      = cat(blocks, prefix)                   // [N + 1, hidden]
//!   score  = rmsnorm(v, no weight) . (norm.weight * proj.weight)
//!   out    = softmax(score) @ v                    // over the RAW candidates
//! ```
//!
//! **It is neither of the two above**, and the difference is structural
//! rather than parametric:
//!
//! 1. the state is one vector PLUS a history of block-boundary
//!    snapshots — not a fixed bundle of parallel streams, and not one
//!    vector alone;
//! 2. the reduce is a softmax over that history against a single learned
//!    score vector (no query, no per-token projection of the state), and
//!    the update is a plain add, not an expansion;
//! 3. the snapshot schedule is periodic in the layer index (`L % B == 0`)
//!    and the exit reduction over the whole history is REQUIRED.
//!
//! A `SingleStream` programme lowers this by discarding the history and
//! every read of it, which computes a different model rather than
//! failing. A `HyperConnection` programme cannot express it at all: no
//! stream count makes a `[1, hidden]` projection a Sinkhorn site's
//! `[(2 + hc)·hc, hc·hidden]` mix, nor a `[hidden]` norm any site
//! operand of one.
//!
//! **The block size is a COMPONENT fact** for the same reason the stream
//! count is: the snapshot schedule, every layer's read of the history,
//! and the stack's own exit reduction all have to agree about it, and a
//! per-layer flag would let one layer snapshot into a history another
//! layer does not read.

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
    /// One residual vector plus a history of block-boundary snapshots of
    /// it; each sublayer READS a softmax-weighted mix over that history
    /// and the current vector, and WRITES by plain addition into the
    /// vector. Kimi-K3's `attn_res_block_size`.
    AttentionResidual {
        /// `attn_res_block_size` — the layer period at which the state
        /// ENTERING a layer is snapshotted into the history (K3 declares
        /// 12 over 93 layers, so eight snapshots exist and the exit
        /// reduction mixes nine candidates).
        ///
        /// Read as declared or not at all. A defaulted block size would
        /// silently change which layers snapshot, which is a different
        /// model rather than a failure — the same reason `hc_mult` is
        /// never defaulted.
        block_size: usize,
    },
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
    ///
    /// One for attention residuals too, and that is a statement rather
    /// than a fallthrough: the prefix sum IS a single vector, and what
    /// the topology adds beside it is a HISTORY of snapshots, whose
    /// length is a function of the position in the stack rather than a
    /// declared width. A caller sizing a buffer from this number gets the
    /// prefix sum right and must ask the topology itself about the
    /// history — which is why the carrier is the traversal transition's
    /// question and not this one's.
    ///
    /// **Decided now, built later**: that carrier's semantic type
    /// encodes a HISTORY (a `[hidden]` prefix beside `[N, hidden]`
    /// snapshots, `N` growing with depth), not streams. It may share the
    /// enter/leave site seam wave 19 built for the bundle, but it is not
    /// a bundle with a different stream count — a bundle's width is
    /// declared once and fixed for the whole stack, and this one is
    /// neither. Reaching for the existing type because it also holds
    /// more than one vector is how a second topology becomes a wrong
    /// dialect of the first.
    pub fn streams(self) -> usize {
        match self {
            Self::SingleStream | Self::AttentionResidual { .. } => 1,
            Self::HyperConnection(hc) => hc.streams,
        }
    }

    /// Whether this is the one-vector residual every family before
    /// hyper-connections carries. Serde reads it to leave a single-stream
    /// plan's serialisation byte-identical to what it was before the
    /// topology travelled on the plan at all.
    ///
    /// `false` for attention residuals even though [`Self::streams`]
    /// answers one: the residual PROGRAMME differs, and a serialisation
    /// that dropped the field would leave a container claiming the
    /// ordinary residual it does not run.
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

    /// **Every declared topology now lowers, and this build has no
    /// `unimplemented_reason` to ask.**
    ///
    /// The refusal that lived here through waves 16-18 was retired in
    /// wave 19; it returned for attention residuals in K3-ATTNRES-1's
    /// first transition, and is retired again here — its readers gone
    /// with it — now that the decode traversal (2a) and the batch
    /// traversal (2b) have each been witnessed against a Torch oracle
    /// transcribed from the reference.
    ///
    /// The function is DELETED rather than left returning `None` for
    /// everything, which is what it did between wave 19 and this rung.
    /// A dead authority that still answers invites a reader to consult
    /// it and conclude something; its own documentation asked that a
    /// variant which refuses again bring the readers back beside it, and
    /// that remains the contract for the next topology.
    ///
    /// What this test can still pin is the part that never depended on
    /// the refusal: an attention residual carries ONE prefix sum, and is
    /// still not the ordinary single-stream residual.
    #[test]
    fn every_declared_topology_lowers_and_attention_residuals_are_not_single_stream() {
        assert_eq!(ResidualTopology::SingleStream.streams(), 1);
        assert!(ResidualTopology::SingleStream.is_single_stream());

        let hc = ResidualTopology::HyperConnection(HyperConnection {
            streams: 4,
            sinkhorn_iters: 20,
            sinkhorn_eps: 1e-6,
        });
        assert_eq!(hc.streams(), 4);
        assert!(!hc.is_single_stream());

        let attn_res = ResidualTopology::AttentionResidual { block_size: 12 };
        // ONE prefix sum, and a history beside it that no width declares.
        assert_eq!(attn_res.streams(), 1);
        // ...and still not the ordinary residual: the programme differs,
        // so a plan carrying it must serialise the field.
        assert!(!attn_res.is_single_stream());
    }

    /// The block size is read as declared and never re-derived: two
    /// components declaring different periods are different topologies,
    /// and the value travels inside the variant rather than beside it.
    #[test]
    fn the_block_size_travels_inside_the_variant() {
        let k3 = ResidualTopology::AttentionResidual { block_size: 12 };
        assert_ne!(k3, ResidualTopology::AttentionResidual { block_size: 6 });
        assert_ne!(k3, ResidualTopology::SingleStream);
        let ResidualTopology::AttentionResidual { block_size } = k3 else {
            panic!("the variant is what carries K3's declared period");
        };
        assert_eq!(block_size, 12);
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
