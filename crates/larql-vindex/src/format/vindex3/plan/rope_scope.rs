//! Which layers a declared RoPE base speaks for.
//!
//! One derivation, shared by the declared-vs-resolved comparator
//! ([`super::compare`]) and the carriage probes ([`super::carriage`]), so
//! the two cannot scope the same declaration differently.
//!
//! A base declared under a per-layer-type section
//! (`rope_parameters.sliding_attention.rope_theta`) speaks for that span.
//! The legacy flat Gemma 3 spelling states the same two facts without a
//! section: `rope_local_base_freq` is the sliding layers' base, and the
//! `rope_theta` beside it is then the full layers' base only.
//! `Gemma3TextConfig.convert_rope_params_to_dict` (transformers 5.5) is the
//! authority: it writes `rope_theta` into `rope_parameters.full_attention`
//! and `rope_local_base_freq` into `rope_parameters.sliding_attention`.
//! Judging that `rope_theta` against every layer — the stack's sliding
//! layers rotate at the local base — is a false mismatch, which is what
//! refused `google/gemma-3-270m-it` and `google/gemma-3-1b-it`.
//!
//! Fail-closed by construction: a family whose resolution ignored
//! `rope_local_base_freq` would rotate its sliding layers at `rope_theta`,
//! and the sliding-span declaration then mismatches and blocks.

use larql_models::inventory::ConfigKeyFact;

use super::super::graph::policy::AttentionSpan;
use super::carriage::ProbeContext;
use super::semantics::{component_of, leaf_of};

/// The leaf every whole-stack and per-section RoPE base is spelled with.
pub const ROPE_THETA_LEAF: &str = "rope_theta";

/// Leaves that name one span's RoPE base without a per-layer-type
/// section around them: `leaf → the span it speaks for`.
pub const SPAN_BASE_SPELLINGS: &[(&str, AttentionSpan)] =
    &[("rope_local_base_freq", AttentionSpan::Sliding)];

/// The scope of one declared RoPE base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeBaseScope {
    /// Speaks for every layer of its component.
    Stack,
    /// Speaks for the layers of one span only.
    Span(AttentionSpan),
    /// A whole-stack spelling another, more specific declaration
    /// overrides; it speaks for no layer on its own.
    Superseded,
}

impl RopeBaseScope {
    /// The span a probe filters by: `None` for the whole stack.
    pub fn probe_span(self) -> Option<AttentionSpan> {
        match self {
            Self::Span(span) => Some(span),
            Self::Stack | Self::Superseded => None,
        }
    }
}

/// Whether `path` declares a RoPE base at all.
pub fn is_rope_base(path: &str) -> bool {
    let leaf = leaf_of(path);
    leaf == ROPE_THETA_LEAF || span_spelling(leaf).is_some()
}

fn span_spelling(leaf: &str) -> Option<AttentionSpan> {
    SPAN_BASE_SPELLINGS
        .iter()
        .find(|(spelling, _)| *spelling == leaf)
        .map(|(_, span)| *span)
}

/// The span a base names by its own path, without consulting siblings.
fn own_span(path: &str) -> Option<AttentionSpan> {
    ProbeContext::span_of(path).or_else(|| span_spelling(leaf_of(path)))
}

/// The layers the RoPE base declared at `path` speaks for.
///
/// A whole-stack spelling narrows to the full span when the same
/// component declares only a sliding-span base, and is superseded when
/// any other per-span base is declared (HF's Gemma 3 lets an explicit
/// `rope_parameters` section win over the flat `rope_theta`).
pub fn rope_base_scope(path: &str, facts: &[ConfigKeyFact]) -> RopeBaseScope {
    if let Some(span) = own_span(path) {
        return RopeBaseScope::Span(span);
    }
    let component = component_of(path);
    let covered: Vec<AttentionSpan> = facts
        .iter()
        .filter(|f| f.path != path && is_rope_base(&f.path) && component_of(&f.path) == component)
        .filter_map(|f| own_span(&f.path))
        .collect();
    if covered.is_empty() {
        RopeBaseScope::Stack
    } else if covered.iter().all(|span| *span == AttentionSpan::Sliding) {
        RopeBaseScope::Span(AttentionSpan::Full)
    } else {
        RopeBaseScope::Superseded
    }
}
