//! Which width a family's ROUTED experts run at, and what the branch
//! does between aggregating them and returning to the residual stream.
//!
//! Most mixture-of-experts blocks run their experts at the model's
//! `hidden_size`: the router picks, the experts consume the block input
//! directly, and their weighted sum is already in the residual stream's
//! space. Kimi-K3 does not. Its routed experts live behind a bottleneck
//! — `routed_expert_down_proj` takes the block input to a narrower
//! `routed_expert_hidden_size`, the experts run THERE, the weighted
//! aggregate is normalised, and `routed_expert_up_proj` returns it to
//! `hidden_size`.
//!
//! Two things stay outside that bottleneck, and getting either wrong
//! computes a different model: the ROUTER reads the un-projected block
//! input, and the SHARED experts do too, being added only after the
//! up-projection.
//!
//! The width the routed experts are SIZED from is derived in exactly one
//! place, and it is not here: `MoeSurface::routed_expert_input_width`
//! answers it for every expert-bank shape contract. This type declares
//! the form; a second `expert_input_width` beside it would be a second
//! authority for one fact, and the two would eventually disagree.
//!
//! [`RoutedExpertForm`] is a separate type rather than a pair of
//! `Option` fields for the same reason [`super::mla::MlaQueryForm`] is:
//! a width and a norm that only exist together should not be
//! independently settable. The reference nests them — `if
//! self.use_latent_moe:` encloses `if self.latent_moe_use_norm:` — so a
//! model declaring a norm and no width builds NO norm, and this type
//! makes that state unrepresentable rather than merely unlikely.

use serde::{Deserialize, Serialize};

/// The epsilon the routed-expert norm runs at.
///
/// Carried explicitly rather than defaulted, because this family's own
/// value is the OPPOSITE of the one the two neighbouring low-rank norms
/// use. `q_a_layernorm` and `kv_a_layernorm` are constructed with no
/// `eps` override and so run at `KimiRMSNorm`'s class default `1e-6`,
/// while `routed_expert_norm` is constructed as
/// `KimiRMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)` and runs
/// at the LAYER's `1e-5`. Two rungs in a row established the class
/// default; assuming a third would be wrong by a factor of ten.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatentNormSpec {
    /// The epsilon, taken from the layer's `rms_norm_eps`.
    pub eps: f64,
}

/// Where a family's routed experts live.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum RoutedExpertForm {
    /// The experts consume the block input at `hidden_size`, and their
    /// weighted sum is already in the residual stream's space. No
    /// wrapper operands exist, and any that ship are refused by name.
    #[default]
    Uniform,
    /// The experts run behind a bottleneck at [`Self::Latent::width`].
    Latent {
        /// The routed experts' input and output width
        /// (`routed_expert_hidden_size`). Not `hidden_size / 2` and not
        /// the expert intermediate width — a third, independently
        /// declared number.
        width: usize,
        /// The norm on the weighted aggregate, between summation and the
        /// up-projection. `None` when the family declares none, which is
        /// a different claim from "an epsilon nobody has established".
        norm: Option<LatentNormSpec>,
    },
}

impl RoutedExpertForm {
    /// Whether this form places a bottleneck around the routed branch,
    /// and therefore requires the wrapper operands.
    pub fn is_latent(self) -> bool {
        matches!(self, RoutedExpertForm::Latent { .. })
    }
}
