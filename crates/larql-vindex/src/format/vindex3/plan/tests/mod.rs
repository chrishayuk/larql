//! Tests for the semantic representability plan.

mod admission;
mod architecture_identity;
mod capability;
mod carriage;
mod compare;
mod gemma3;
mod gemma3_text_flat;
mod gemma4;
mod hybrid_linear_attention;
mod identity;
mod k3_representable;
mod mla_nope;
mod moe_spellings;
mod plan_resolved;
mod qw35d_admission;
mod recurrence_identification;
mod registration_grants_nothing;
mod relative_position;
mod rope_scope;
mod semantics;
mod situ;
mod system;

/// Fixtures live one level up so the graph tests share them.
use super::tests_support as support;
