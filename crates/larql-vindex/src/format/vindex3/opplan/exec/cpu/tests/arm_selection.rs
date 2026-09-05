//! The process-wide execution arms, resolved from their variables by pure
//! functions — so BOTH answers of each arm are testable in one process,
//! which a `OnceLock` read of the environment never allowed.
//!
//! Every arm follows one rule: the exact word selects, anything else is
//! the default. A typo must not invent a numerical regime and then be
//! reported as a measurement.

use super::super::physical::{ArithmeticArm, Q4Classes};
use crate::format::vindex3::opplan::exec::backend::MatrixClass;

#[test]
fn unset_empty_and_all_are_the_blanket_q4_arm() {
    for v in [None, Some(""), Some("  "), Some("all"), Some(" all\n")] {
        assert_eq!(Q4Classes::from_env_value(v), Q4Classes::ALL, "{v:?}");
    }
}

#[test]
fn a_q4_class_set_names_exactly_the_classes_it_lists() {
    let set = Q4Classes::from_env_value(Some("attn, ffn"));
    assert!(set.attention && set.ffn && !set.head, "{set:?}");
    let head = Q4Classes::from_env_value(Some("head"));
    assert!(!head.attention && !head.ffn && head.head, "{head:?}");
    // Tokens are exact: a wrong case names nothing, which restores every
    // class to Q8 rather than silently admitting all of them.
    let none = Q4Classes::from_env_value(Some("ATTN,FFN,HEAD"));
    assert!(!none.attention && !none.ffn && !none.head, "{none:?}");
}

#[test]
fn admission_follows_the_set_and_the_bank_is_never_admitted() {
    let all = Q4Classes::ALL;
    assert!(all.admits(MatrixClass::AttentionProjection));
    assert!(all.admits(MatrixClass::FfnProjection));
    assert!(all.admits(MatrixClass::OutputHead));
    assert!(
        !all.admits(MatrixClass::RoutedExpertBank),
        "the bank is widened on the way in; no compact bytes remain to admit"
    );
    let ffn_only = Q4Classes::from_env_value(Some("ffn"));
    assert!(!ffn_only.admits(MatrixClass::AttentionProjection));
    assert!(ffn_only.admits(MatrixClass::FfnProjection));
    assert!(!ffn_only.admits(MatrixClass::OutputHead));
}

#[test]
fn each_arithmetic_arm_answers_to_its_two_spellings_and_nothing_else() {
    use ArithmeticArm::*;
    assert_eq!(ArithmeticArm::from_env_value(None), FloatActivation);
    assert_eq!(ArithmeticArm::default(), FloatActivation);
    for (v, want) in [
        ("bf16xq8", Bf16TimesQ8),
        ("bf16xq8b", Bf16TimesQ8),
        ("q8xq8", Q8TimesQ8),
        (" q8xq8b\n", Q8TimesQ8),
        ("q4xq8", Q4TimesQ8),
        ("q4xq8b", Q4TimesQ8),
    ] {
        assert_eq!(ArithmeticArm::from_env_value(Some(v)), want, "{v:?}");
    }
    for v in ["", "Q8XQ8", "q8", "q8xq8bb", "fp32", "nonsense"] {
        assert_eq!(
            ArithmeticArm::from_env_value(Some(v)),
            FloatActivation,
            "{v:?} must be the default, not a fourth regime"
        );
    }
}
