//! Tests for the semantic representability plan.

mod architecture_identity;
mod capability;
mod carriage;
mod compare;
mod gemma4;
mod hybrid_linear_attention;
mod identity;
mod k3_representable;
mod mla_nope;
mod moe_spellings;
mod qw35d_admission;
mod recurrence_identification;
mod registration_grants_nothing;
mod relative_position;
mod semantics;
mod system;

/// Fixtures live one level up so the graph tests share them.
use super::tests_support as support;
