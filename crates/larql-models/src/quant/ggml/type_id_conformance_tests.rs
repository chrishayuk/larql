//! Every ggml type id this crate names, pinned against ggml's own table.
//!
//! ## The bug this exists to prevent returning
//!
//! `TYPE_Q8_0` was `6` and `TYPE_Q5_0` was `8`. Upstream ggml has
//! `Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9` — the four-id legacy block
//! was transposed. `loading/gguf/parser.rs` reads a tensor's type id
//! straight out of the file and hands it to [`super::dequantize`], so a
//! GGUF carrying a genuine Q8_0 tensor was decoded as Q5_0: wrong values
//! *and* a wrong block stride (22 bytes read where 34 were written).
//! Q8_0 is one of the most common GGUF quantisations.
//!
//! ## Why nothing caught it
//!
//! Every caller inside this workspace passes these same constants in
//! both directions — encode with `TYPE_Q8_0`, decode with `TYPE_Q8_0` —
//! so a transposed pair cancels and every round-trip test passes. The
//! error is only observable where the value crosses to another
//! implementation, and until a K-quant encode was routed through ggml
//! itself, it never did.
//!
//! > **Internal agreement is not evidence of external correctness.**
//!
//! A byte layout has the same property, and is guarded the same way by
//! `larql-vindex`'s `ggml_kquant_golden` fixture. This is that guard for
//! the *identifiers*.
//!
//! ## The fixture
//!
//! `fixtures/ggml_type_table.json` is dumped from `ggml_get_type_traits`
//! by `fixtures/ggml_type_table.gen.c` (committed beside it). It is not
//! transcribed from a header, because a transcription is exactly the
//! step that went wrong.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Deserialize)]
struct Table {
    types: Vec<TypeRow>,
}

#[derive(Deserialize)]
struct TypeRow {
    id: u32,
    name: String,
    blck_size: u64,
    type_size: u64,
}

fn upstream() -> BTreeMap<String, TypeRow> {
    let table: Table = serde_json::from_str(include_str!("fixtures/ggml_type_table.json"))
        .expect("the ggml type table parses");
    table
        .types
        .into_iter()
        .filter(|t| t.name != "DEPRECATED")
        .map(|t| (t.name.to_ascii_lowercase(), t))
        .collect()
}

/// Every constant this crate defines for a type ggml also defines, with
/// the ggml spelling it corresponds to.
///
/// Listed explicitly rather than derived: the mapping from a LARQL
/// constant to an upstream name is precisely the thing that was wrong,
/// so it has to be stated where a reader can check it, not inferred by a
/// rule that could be wrong in the same direction.
fn named() -> Vec<(&'static str, u32, &'static str)> {
    use super::*;
    vec![
        ("TYPE_F32", TYPE_F32, "f32"),
        ("TYPE_F16", TYPE_F16, "f16"),
        ("TYPE_Q4_0", TYPE_Q4_0, "q4_0"),
        ("TYPE_Q4_1", TYPE_Q4_1, "q4_1"),
        ("TYPE_Q5_0", TYPE_Q5_0, "q5_0"),
        ("TYPE_Q5_1", TYPE_Q5_1, "q5_1"),
        ("TYPE_Q8_0", TYPE_Q8_0, "q8_0"),
        ("TYPE_Q8_1", TYPE_Q8_1, "q8_1"),
        ("TYPE_Q2_K", TYPE_Q2_K, "q2_K"),
        ("TYPE_Q3_K", TYPE_Q3_K, "q3_K"),
        ("TYPE_Q4_K", TYPE_Q4_K, "q4_K"),
        ("TYPE_Q5_K", TYPE_Q5_K, "q5_K"),
        ("TYPE_Q6_K", TYPE_Q6_K, "q6_K"),
        ("TYPE_BF16", TYPE_BF16, "bf16"),
    ]
}

#[test]
fn every_named_type_id_matches_upstream_ggml() {
    let up = upstream();
    let mut wrong = Vec::new();
    for (konst, id, ggml_name) in named() {
        let Some(row) = up.get(&ggml_name.to_ascii_lowercase()) else {
            panic!("ggml has no type named `{ggml_name}` — the fixture or the mapping is stale");
        };
        if row.id != id {
            wrong.push(format!(
                "{konst} = {id} but ggml's `{ggml_name}` is {} (a GGUF tensor of this type \
                 would be decoded as whatever id {id} actually is)",
                row.id
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "type ids disagree with upstream: {wrong:#?}"
    );
}

/// The transposition specifically, named so a regression reads as the
/// historical bug rather than as an anonymous number change.
#[test]
fn the_legacy_block_is_not_transposed_again() {
    use super::{TYPE_Q5_0, TYPE_Q5_1, TYPE_Q8_0, TYPE_Q8_1};
    assert_eq!(TYPE_Q5_0, 6, "Q5_0 is 6 upstream; it was once 8 here");
    assert_eq!(TYPE_Q5_1, 7, "Q5_1 is 7 upstream; it was once 9 here");
    assert_eq!(TYPE_Q8_0, 8, "Q8_0 is 8 upstream; it was once 6 here");
    assert_eq!(TYPE_Q8_1, 9, "Q8_1 is 9 upstream");
    // All four distinct — a copy-paste that collapsed two of them would
    // otherwise satisfy the individual assertions above in isolation.
    let ids = [TYPE_Q5_0, TYPE_Q5_1, TYPE_Q8_0, TYPE_Q8_1];
    let distinct: std::collections::BTreeSet<u32> = ids.into_iter().collect();
    assert_eq!(distinct.len(), ids.len(), "the legacy ids are not distinct");
}

/// The decoders' block geometry, checked against ggml rather than
/// against this crate's own decoders — which is where the numbers were
/// read from, so they cannot independently confirm them.
#[test]
fn the_block_geometry_this_crate_decodes_matches_upstream() {
    let up = upstream();
    for (name, blck, size) in [
        ("q4_0", 32u64, 18u64),
        ("q4_1", 32, 20),
        ("q5_0", 32, 22),
        ("q5_1", 32, 24),
        ("q8_0", 32, 34),
        ("q2_k", 256, 84),
        ("q3_k", 256, 110),
        ("q4_k", 256, 144),
        ("q5_k", 256, 176),
        ("q6_k", 256, 210),
    ] {
        let row = up
            .get(name)
            .unwrap_or_else(|| panic!("ggml has no `{name}`"));
        assert_eq!(row.blck_size, blck, "{name}: block size");
        assert_eq!(row.type_size, size, "{name}: bytes per block");
    }
}

/// The fixture must be capable of failing: if it were empty, or if every
/// id were its index, the checks above would pass vacuously.
#[test]
fn the_fixture_is_a_real_table() {
    let up = upstream();
    assert!(
        up.len() > 20,
        "only {} types — the fixture looks truncated",
        up.len()
    );
    // ggml's table has gaps (ids 4 and 5 are deprecated), so a table
    // whose ids were simply 0..n would not be ggml's.
    assert!(
        up.values().any(|t| t.id > up.len() as u32),
        "no id exceeds the row count — this table has no gaps and so is not ggml's"
    );
}
