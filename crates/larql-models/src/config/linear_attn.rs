//! The KDA block's declared geometry, and the spelling used for a layer
//! whose declared interleave could not be read.
//!
//! The interleave itself lives in [`super::interleave`] — one abstraction
//! over every spelling — rather than here, where it was specific to
//! `linear_attn_config`.

use serde::{Deserialize, Serialize};

/// `layer_types` spelling for a layer whose interleave was **declared but
/// could not be resolved**.
///
/// Deliberately outside every recognised vocabulary, so it resolves to no
/// span, fails `matches_declaration`, and blocks. The alternative — leaving
/// the per-layer declaration absent — would let the caller's default answer
/// for a topology the checkpoint actually stated and this build failed to
/// read, which is the silent degradation this module exists to prevent.
pub const LAYER_TYPE_UNRESOLVED_INTERLEAVE: &str = "declared-interleave(unresolved)";

/// The KDA block's declared geometry — the second half of what
/// `linear_attn_config` states, beside the interleave.
///
/// Separate from
/// [`LinearAttentionTopology`](crate::inventory::LinearAttentionTopology),
/// which reads Qwen3.8's `linear_*` spellings and describes Gated DeltaNet.
/// The two operators are not variants of one another: Gated DeltaNet
/// carries a *fused* q|k|v projection, one conv, full-rank gates and a
/// per-**head** `dt_bias`, while KDA carries split projections, three
/// convs, low-rank f/g gates and a per-**channel** `dt_bias`. Keeping the
/// geometries apart is what stops one being read as the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KdaGeometry {
    /// `num_heads` — Hv. 64 on GLM-5.3-Flash, 32 on Kimi Linear.
    pub num_heads: usize,
    /// `head_dim` — Dk = Dv. 128 on both observed checkpoints, but read,
    /// never assumed.
    pub head_dim: usize,
    /// `short_conv_kernel_size` — width of the depthwise causal conv each
    /// of q, k and v passes through independently.
    pub conv_kernel: usize,
}

impl KdaGeometry {
    /// Read the geometry from a `linear_attn_config` section.
    ///
    /// All three or none: a partial declaration is refused rather than
    /// completed with defaults, the same contract
    /// `LinearAttentionTopology::from_config` holds for Gated DeltaNet.
    pub fn read(section: &serde_json::Value) -> Option<Self> {
        let field = |key: &str| section[key].as_u64().map(|v| v as usize).filter(|v| *v > 0);
        Some(Self {
            num_heads: field("num_heads")?,
            head_dim: field("head_dim")?,
            conv_kernel: field("short_conv_kernel_size")?,
        })
    }

    /// Width of the value/gate side, `Hv·Dv` — the row count of each of
    /// the three projections, of each conv, and the length of `dt_bias`.
    ///
    /// This is the number that separates KDA from Gated DeltaNet at a
    /// glance: DeltaNet's `dt_bias` is `[Hv]`, KDA's is `[Hv·Dv]`.
    pub fn value_width(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements in one layer's recurrent state: a `Dk × Dv` matrix per
    /// head.
    pub fn state_elements(self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }
}
