//! Compatibility fixtures for the persisted representation vocabulary.
//!
//! Split from `tests.rs` so the guard on what is ON DISK is easy to find
//! and hard to edit casually — it is the only thing standing between a
//! rename for prettier output and a precision map that no longer
//! resolves.

use super::*;

/// **The persisted spellings of every role, frozen.**
///
/// These strings are already on disk and on the wire: `PrecisionMap`
/// stores `roles` as text, and `--include-role` takes them from users.
/// A precision map written today has to keep resolving after any future
/// edit to this crate, so role IDENTITY is a compatibility surface even
/// though the enum is internal.
///
/// The invariant this pins: **display wording may evolve; persisted
/// role identity does not.** Today `Role::name` serves both jobs, which
/// is fine precisely because this test stands in the way — anyone
/// renaming it for prettier CLI output fails here and is forced to
/// decide. When the two genuinely need to diverge, the move is to add a
/// `wire_name` that keeps these strings and let `name` become
/// human-facing; this fixture then governs `wire_name`, unchanged.
const GOLDEN_ROLE_WIRE_NAMES: &[(&str, policy::Role)] = &[
    ("decoder-linear", policy::Role::DecoderLinear),
    ("recurrence-projection", policy::Role::RecurrenceProjection),
    ("recurrence-control", policy::Role::RecurrenceControl),
    ("expert-weight", policy::Role::ExpertWeight),
    ("embedding", policy::Role::Embedding),
    ("output-head", policy::Role::OutputHead),
    ("norm", policy::Role::Norm),
    ("router", policy::Role::Router),
    ("small-vector", policy::Role::SmallVector),
    ("auxiliary-component", policy::Role::AuxiliaryComponent),
    ("unknown", policy::Role::Unknown),
];

#[test]
fn the_golden_role_spellings_still_deserialize() {
    for (wire, want) in GOLDEN_ROLE_WIRE_NAMES {
        let got: policy::Role =
            serde_json::from_str(&format!("\"{wire}\"")).unwrap_or_else(|e| {
                panic!("`{wire}` no longer deserialises ({e}) — a precision map on disk that names it can no longer be read")
            });
        assert_eq!(got, *want, "`{wire}` now means a different role");
        assert_eq!(
            policy::Role::parse(wire),
            Some(*want),
            "`{wire}` is still accepted on the wire but no longer by --include-role"
        );
        assert_eq!(
            want.name(),
            *wire,
            "`{want}` renamed its persisted identity"
        );
    }
}

/// A NEW role must be given a persisted spelling deliberately, not
/// inherit one from however the variant happens to be named.
#[test]
fn every_role_has_a_golden_spelling() {
    assert_eq!(
        GOLDEN_ROLE_WIRE_NAMES.len(),
        policy::Role::ALL.len(),
        "a role was added or removed without deciding its persisted spelling — add it to \
         GOLDEN_ROLE_WIRE_NAMES, and remember that removing one orphans every precision map \
         that names it"
    );
    for r in policy::Role::ALL {
        assert!(
            GOLDEN_ROLE_WIRE_NAMES.iter().any(|(_, g)| g == r),
            "{r} has no golden spelling"
        );
    }
}

/// A precision map written with the golden spellings still governs the
/// tensors it was written for — the end-to-end form of the guarantee,
/// since `PrecisionMap` matches roles by string.
#[test]
fn a_precision_map_written_with_golden_spellings_still_resolves() {
    let json = r#"{
        "name": "golden-compat",
        "encoding": "Q6_K",
        "roles": ["decoder-linear", "expert-weight"],
        "exceptions": [{"projection": "down_proj", "layers": [20, 26]}]
    }"#;
    let map: map::PrecisionMap = serde_json::from_str(json).expect("golden map parses");
    assert!(matches!(
        map.resolve(
            policy::Role::ExpertWeight,
            "3.mlp.experts.7.gate_proj.weight"
        ),
        map::Precision::Compiled("Q6_K")
    ));
    // The exception still carves out its region, and a role the map does
    // not name still falls back to source precision.
    assert!(matches!(
        map.resolve(
            policy::Role::ExpertWeight,
            "22.mlp.experts.7.down_proj.weight"
        ),
        map::Precision::Source
    ));
    assert!(matches!(
        map.resolve(policy::Role::Router, "3.mlp.gate.weight"),
        map::Precision::Source
    ));
}
