//! The [`ModelArchitecture`] trait — what a model *is*, with no compute.
//!
//! The trait is a stack of concept supertraits, one file each, each extending
//! the one before it:
//!
//! ```text
//! ArchitectureCore → TensorKeys → Norms → Position → Attention
//!   → FeedForward → LatentAttention → Embeddings → ModelArchitecture
//! ```
//!
//! Callers name only `ModelArchitecture` (or `dyn ModelArchitecture`):
//! supertrait methods resolve through it. An implementor writes one `impl`
//! per supertrait, empty where it keeps every default; code holding a
//! concrete architecture type imports [`prelude`].
//!
//! Everything a default body needs in order to be short lives outside the
//! trait: selector constants ([`super::rope_types`], [`super::layer_types`]),
//! parameter structs ([`super::rope`]) and the rules that read them. A
//! default should read as one config fact, one decision.
//!
//! **Default bodies that read the config are the point, not a shortcut.**
//! `docs/k3-funnel.md` §4.7.8 records the same bug three times over: a
//! behaviour that is a config fact, implemented on one architecture, with a
//! trait default silently answering for every other. A `None`-returning
//! default is safe only when the answer genuinely is "this family does not
//! have one"; where `config.json` states the answer, read it here.

mod attention;
mod embeddings;
mod feed_forward;
mod identity;
mod latent_attention;
mod model_architecture;
mod norms;
mod position;
mod shared;
mod tensor_keys;

pub use attention::Attention;
pub use attention::UNSCALED_SCORES;
pub use embeddings::Embeddings;
pub use feed_forward::FeedForward;
pub use identity::ArchitectureCore;
pub use latent_attention::LatentAttention;
pub use model_architecture::ModelArchitecture;
pub use norms::Norms;
pub use position::{default_position_policy_for_layer, Position};
pub use shared::{score_scale_from_query_pre_attn_scalar, UNSCALED_POSITION_DIVISOR};
use shared::{IDENTITY_SCALE, SITU_DEFAULT_BETA};
pub use tensor_keys::TensorKeys;

/// Every supertrait of [`ModelArchitecture`], for implementors and for
/// callers holding a concrete architecture type rather than `dyn`.
pub mod prelude {
    pub use super::{
        ArchitectureCore, Attention, Embeddings, FeedForward, LatentAttention, ModelArchitecture,
        Norms, Position, TensorKeys,
    };
}
