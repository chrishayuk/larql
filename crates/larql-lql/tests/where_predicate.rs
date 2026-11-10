//! `WHERE` clauses are validated as a unit before any verb acts on them.
//!
//! Every clause below used to be *accepted*: the executor kept the
//! conditions it understood and ignored the rest, so an unknown field, a
//! wrong-typed value or an unsupported operator widened the predicate —
//! all the way to "every feature" when nothing was left.
//!
//! The contract these gates pin, identically for SELECT, DELETE and
//! UPDATE:
//!
//! - a clause with any condition the executor cannot evaluate is refused
//!   whole, including when other conditions in it are valid;
//! - a refused DELETE/UPDATE changes nothing: the rows SELECT shows are
//!   the same and the patch recording gains no operation;
//! - a predicate that *is* supported narrows — it never matches more than
//!   it says.

use std::collections::BTreeSet;
use std::path::Path;

use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, FeatureMeta, QuantFormat, StorageDtype, VectorIndex, VindexConfig,
};

/// The phrase every predicate refusal carries.
const REFUSAL: &str = "unsupported WHERE clause";
/// A LIMIT above any fixture's row count, so SELECT shows everything.
const ALL_ROWS: u32 = 1_000_000;

// The fixture: two layers of three feature slots. Layer 1's middle slot
// carries no metadata, so the vindex holds five edges.
const HIDDEN: usize = 4;
const FEATURES_PER_LAYER: usize = 3;
const VOCAB: usize = 10;
/// `(top token, score)` per slot; `None` is a slot with no edge.
const LAYERS: [[Option<(&str, f32)>; FEATURES_PER_LAYER]; 2] = [
    [
        Some(("Paris", 0.95)),
        Some(("French", 0.88)),
        Some(("Europe", 0.75)),
    ],
    [Some(("Berlin", 0.90)), None, Some(("Spain", 0.70))],
];
const EDGE_COUNT: usize = 5;
/// A token present in every slot's top-k and in no slot's top token.
const ALSO_TOKEN: &str = "France";
const ALSO_TOKEN_ID: u32 = 9;
/// A threshold two of the five edges fall under (0.75 and 0.70).
const LOW_SCORE: &str = "0.8";
const LOW_SCORE_VALUE: f32 = 0.8;
const EDGES_BELOW_LOW_SCORE: usize = 2;

/// Clauses no verb evaluates. Each one silently widened before.
const REFUSED_CLAUSES: &[(&str, &str)] = &[
    ("unknown field", r#"colour = "red""#),
    ("string where a layer index belongs", r#"layer = "oops""#),
    ("integer where an entity belongs", "entity = 5"),
    ("inequality on a layer", "layer > 0"),
    ("inequality on a feature", "feature <= 3"),
    ("negated entity", r#"entity != "France""#),
    ("entity list", r#"entity IN ("France", "Germany")"#),
    (
        "range written as two conditions",
        "layer >= 0 AND layer <= 1",
    ),
    ("negative feature index", "layer = 0 AND feature = -1"),
    (
        "valid condition beside an unknown field",
        r#"layer = 0 AND colour = "red""#,
    ),
    (
        "valid condition beside a wrong type",
        r#"feature = 0 AND layer = "oops""#,
    ),
    ("ordering operator on a relation", r#"relation > "capital""#),
    ("string where a score belongs", r#"confidence = "high""#),
    ("empty entity, which every token contains", r#"entity = """#),
    (
        "empty relation, which every label contains",
        r#"relation = """#,
    ),
];

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse {stmt}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    try_run(session, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

/// A session bound to a synthetic vindex, recording into a named patch.
struct Fixture {
    session: Session,
    patch: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

/// The vindex stores token ids; the strings come back through this
/// tokenizer, so it must name every id the fixture uses.
fn tokenizer_json() -> String {
    let mut vocab: Vec<String> = Vec::new();
    for (layer, slots) in LAYERS.iter().enumerate() {
        for (feature, slot) in slots.iter().enumerate() {
            if let Some((token, _)) = slot {
                vocab.push(format!("\"{token}\":{}", slot_token_id(layer, feature)));
            }
        }
    }
    vocab.push(format!("\"{ALSO_TOKEN}\":{ALSO_TOKEN_ID}"));
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"{ALSO_TOKEN}\"}}}}",
        vocab.join(",")
    )
}

fn slot_token_id(layer: usize, feature: usize) -> u32 {
    (layer * FEATURES_PER_LAYER + feature) as u32
}

/// Write a browse-level vindex whose feature metadata is [`LAYERS`].
fn write_edge_vindex(dir: &Path) {
    let mut gates = Vec::new();
    let mut down_meta = Vec::new();
    for (layer, slots) in LAYERS.iter().enumerate() {
        let mut gate = Array2::<f32>::zeros((FEATURES_PER_LAYER, HIDDEN));
        let mut metas = Vec::new();
        for (feature, slot) in slots.iter().enumerate() {
            gate[[feature, (layer + feature) % HIDDEN]] = 1.0;
            metas.push(slot.map(|(token, score)| {
                let token_id = slot_token_id(layer, feature);
                FeatureMeta {
                    top_token: token.to_string(),
                    top_token_id: token_id,
                    c_score: score,
                    // Every slot also lists `ALSO_TOKEN` further down its
                    // top-k: an entity that is never a top token.
                    top_k: vec![
                        TopKEntry {
                            token: token.to_string(),
                            token_id,
                            logit: score,
                        },
                        TopKEntry {
                            token: ALSO_TOKEN.to_string(),
                            token_id: ALSO_TOKEN_ID,
                            logit: score / 2.0,
                        },
                    ],
                }
            }));
        }
        gates.push(Some(gate));
        down_meta.push(Some(metas));
    }

    let index = VectorIndex::new(gates, down_meta, LAYERS.len(), HIDDEN);
    let mut config = VindexConfig {
        version: 2,
        model: "test/where-predicate".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers: LAYERS.len(),
        hidden_size: HIDDEN,
        intermediate_size: FEATURES_PER_LAYER,
        vocab_size: VOCAB,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(dir, &mut config).expect("save vindex");
    std::fs::write(
        dir.join("embeddings.bin"),
        vec![0u8; VOCAB * HIDDEN * std::mem::size_of::<f32>()],
    )
    .expect("embeddings");
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).expect("tokenizer");
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let vindex = dir.path().join("source.vindex");
    std::fs::create_dir_all(&vindex).expect("vindex dir");
    write_edge_vindex(&vindex);
    let patch = dir.path().join("recording.vlp");

    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&vindex)));
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch)),
    );
    Fixture {
        session,
        patch,
        _dir: dir,
    }
}

impl Fixture {
    /// Every edge row the session currently shows.
    fn rows(&mut self) -> Vec<String> {
        run(
            &mut self.session,
            &format!("SELECT * FROM EDGES LIMIT {ALL_ROWS};"),
        )
    }

    /// How many edge rows the session still shows (`L<layer>  F<feature> …`).
    fn edge_count(&mut self) -> usize {
        self.rows()
            .iter()
            .filter(|row| {
                let mut chars = row.chars();
                chars.next() == Some('L') && chars.next().is_some_and(|c| c.is_ascii_digit())
            })
            .count()
    }

    /// Save the recording and count the operations it holds.
    fn recorded_operations(mut self) -> usize {
        run(&mut self.session, "SAVE PATCH;");
        let text = std::fs::read_to_string(&self.patch).expect("saved patch");
        let json: serde_json::Value = serde_json::from_str(&text).expect("patch json");
        json["operations"].as_array().expect("operations").len()
    }
}

fn assert_refused(result: Result<Vec<String>, String>, stmt: &str) {
    let err = match result {
        Ok(out) => panic!("{stmt}: expected a refusal, statement ran: {out:?}"),
        Err(e) => e,
    };
    assert!(err.contains(REFUSAL), "{stmt}: unexpected error: {err}");
}

// ── Refusals, identical across the three verbs ───────────────────────────

#[test]
fn select_refuses_every_unevaluable_clause() {
    let mut fx = fixture();
    for (what, clause) in REFUSED_CLAUSES {
        let stmt = format!("SELECT * FROM EDGES WHERE {clause};");
        assert_refused(try_run(&mut fx.session, &stmt), &format!("{what}: {stmt}"));
    }
}

#[test]
fn delete_refuses_every_unevaluable_clause_and_changes_nothing() {
    let mut fx = fixture();
    let before = fx.rows();
    assert_eq!(fx.edge_count(), EDGE_COUNT, "the fixture has edges to lose");

    for (what, clause) in REFUSED_CLAUSES {
        let stmt = format!("DELETE FROM EDGES WHERE {clause};");
        assert_refused(try_run(&mut fx.session, &stmt), &format!("{what}: {stmt}"));
        assert_eq!(fx.rows(), before, "{what}: the overlay changed");
    }
    assert_eq!(fx.recorded_operations(), 0, "a refused DELETE was recorded");
}

#[test]
fn update_refuses_every_unevaluable_clause_and_changes_nothing() {
    let mut fx = fixture();
    let before = fx.rows();

    for (what, clause) in REFUSED_CLAUSES {
        let stmt = format!("UPDATE EDGES SET target = \"changed\" WHERE {clause};");
        assert_refused(try_run(&mut fx.session, &stmt), &format!("{what}: {stmt}"));
        assert_eq!(fx.rows(), before, "{what}: the overlay changed");
    }
    assert_eq!(fx.recorded_operations(), 0, "a refused UPDATE was recorded");
}

// ── Supported predicates narrow; they never widen ────────────────────────

#[test]
fn delete_by_score_removes_only_the_edges_under_the_threshold() {
    // The headline reproducer: a score predicate on DELETE used to be
    // dropped, tombstoning every feature in the vindex.
    let mut fx = fixture();

    let out = run(
        &mut fx.session,
        &format!("DELETE FROM EDGES WHERE confidence < {LOW_SCORE};"),
    );

    assert!(
        out.join("\n")
            .contains(&format!("Deleted {EDGES_BELOW_LOW_SCORE} features")),
        "{out:?}"
    );
    assert_eq!(fx.edge_count(), EDGE_COUNT - EDGES_BELOW_LOW_SCORE);
    assert_eq!(fx.recorded_operations(), EDGES_BELOW_LOW_SCORE);
}

#[test]
fn update_by_score_rewrites_only_the_edges_under_the_threshold() {
    let mut fx = fixture();

    let out = run(
        &mut fx.session,
        &format!("UPDATE EDGES SET target = \"changed\" WHERE score < {LOW_SCORE};"),
    );

    assert!(
        out.join("\n")
            .contains(&format!("Updated {EDGES_BELOW_LOW_SCORE} features")),
        "{out:?}"
    );
    assert_eq!(
        fx.edge_count(),
        EDGE_COUNT,
        "UPDATE relabels; it removes nothing"
    );
    assert_eq!(fx.recorded_operations(), EDGES_BELOW_LOW_SCORE);
}

#[test]
fn select_by_score_shows_only_the_edges_under_the_threshold() {
    let mut fx = fixture();

    let rows = run(
        &mut fx.session,
        &format!("SELECT * FROM EDGES WHERE confidence < {LOW_SCORE} LIMIT {ALL_ROWS};"),
    )
    .join("\n");

    for (_, score) in LAYERS.iter().flatten().flatten() {
        let shown = rows.contains(&format!("{score:.4}"));
        assert_eq!(shown, *score < LOW_SCORE_VALUE, "score {score}: {rows}");
    }
}

#[test]
fn a_pinned_slot_with_an_entity_it_does_not_hold_matches_nothing() {
    // `layer` + `feature` used to short-circuit past the entity check.
    let mut fx = fixture();

    let out = run(
        &mut fx.session,
        r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0 AND entity = "Berlin";"#,
    );

    assert!(out.join("\n").contains("no matching features"), "{out:?}");
    assert_eq!(fx.edge_count(), EDGE_COUNT);
    assert_eq!(fx.recorded_operations(), 0);
}

#[test]
fn delete_of_a_pinned_slot_records_exactly_that_slot() {
    let mut fx = fixture();

    let out = run(
        &mut fx.session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );

    assert!(out.join("\n").contains("Deleted 1 features"), "{out:?}");
    assert_eq!(fx.recorded_operations(), 1);
}

#[test]
fn a_pinned_slot_outside_the_vindex_is_refused_not_recorded() {
    // `layer` and `feature` both pinned used to skip every check: a slot
    // that does not exist was "deleted" and written into the patch.
    let mut fx = fixture();
    let before = fx.rows();

    for clause in [
        "layer = 0 AND feature = 999999999",
        "layer = 1 AND feature = 3",
        "layer = 999999 AND feature = 0",
    ] {
        for stmt in [
            format!("DELETE FROM EDGES WHERE {clause};"),
            format!("UPDATE EDGES SET target = \"changed\" WHERE {clause};"),
        ] {
            let err = try_run(&mut fx.session, &stmt).expect_err(&stmt);
            assert!(err.contains("no such feature slot"), "{stmt}: {err}");
            assert_eq!(fx.rows(), before, "{stmt}: the overlay changed");
        }
    }
    assert_eq!(fx.recorded_operations(), 0);
}

#[test]
fn a_select_without_a_where_clause_shows_every_edge() {
    let mut fx = fixture();
    assert_eq!(fx.edge_count(), EDGE_COUNT);
}

// ── Session state is untouched by a refusal ──────────────────────────────

/// A session with no `BEGIN PATCH`: the first successful mutation opens
/// an anonymous auto-patch, and nothing else may.
fn session_without_a_patch() -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let vindex = dir.path().join("source.vindex");
    std::fs::create_dir_all(&vindex).expect("vindex dir");
    write_edge_vindex(&vindex);
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&vindex)));
    (session, dir)
}

#[test]
fn a_refused_or_matchless_mutation_does_not_open_the_auto_patch() {
    // `SAVE PATCH` distinguishes "no patch session" from "anonymous
    // session": only the former means nothing was started.
    const NO_SESSION: &str = "no active patch session";
    let (mut session, _dir) = session_without_a_patch();

    for stmt in [
        r#"DELETE FROM EDGES WHERE colour = "red";"#,
        r#"UPDATE EDGES SET target = "x" WHERE layer > 0;"#,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 999999999;",
        "DELETE FROM EDGES WHERE confidence > 1000000;",
    ] {
        let _ = try_run(&mut session, stmt);
        let err = try_run(&mut session, "SAVE PATCH;").expect_err("nothing was started");
        assert!(
            err.contains(NO_SESSION),
            "{stmt}: a patch session exists: {err}"
        );
    }

    let out = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    assert!(out.join("\n").contains("Auto-patch started"), "{out:?}");
}

// ── The other consumers of a WHERE clause ────────────────────────────────

#[test]
fn select_nearest_refuses_a_where_clause_it_would_ignore() {
    let mut fx = fixture();
    let stmt = r#"SELECT * FROM EDGES NEAREST TO "Paris" AT LAYER 0 WHERE score > 0.9;"#;
    assert_refused(try_run(&mut fx.session, stmt), stmt);
}

#[test]
fn show_features_honours_the_score_operator_and_refuses_the_rest() {
    let mut fx = fixture();

    let low = run(
        &mut fx.session,
        &format!("SHOW FEATURES 0 WHERE confidence < {LOW_SCORE};"),
    )
    .join("\n");
    // Layer 0 holds scores 0.95, 0.88 and 0.75; only the last is under.
    assert!(low.contains("0.7500"), "{low}");
    assert!(!low.contains("0.9500") && !low.contains("0.8800"), "{low}");

    for clause in [r#"colour = "red""#, r#"relation = "capital""#, "layer = 0"] {
        let stmt = format!("SHOW FEATURES 0 WHERE {clause};");
        assert_refused(try_run(&mut fx.session, &stmt), &stmt);
    }
}

#[test]
fn select_from_features_and_entities_refuse_fields_they_do_not_have() {
    let mut fx = fixture();
    for stmt in [
        "SELECT * FROM FEATURES WHERE score > 0.5;",
        r#"SELECT * FROM FEATURES WHERE layer = "oops";"#,
        "SELECT * FROM ENTITIES WHERE feature = 1;",
        "SELECT * FROM ENTITIES WHERE layer > 0;",
    ] {
        assert_refused(try_run(&mut fx.session, stmt), stmt);
    }
}

// ── SELECT lists exactly what DELETE and UPDATE target ───────────────────
//
// For the same vindex, clause and starting overlay, the slots SELECT
// lists (unpaginated) must be the slots a mutation records. One resolver
// serves all three; these gates hold it to that from the outside.

type Slots = BTreeSet<(u64, u64)>;

/// `(layer, feature)` of every row a SELECT printed: `L<layer>  F<feature> …`.
fn listed_slots(rows: &[String]) -> Slots {
    rows.iter()
        .filter_map(|row| {
            let mut columns = row.split_whitespace();
            let layer = columns.next()?.strip_prefix('L')?.parse().ok()?;
            let feature = columns.next()?.strip_prefix('F')?.parse().ok()?;
            Some((layer, feature))
        })
        .collect()
}

/// `(layer, feature)` of every operation in a saved patch.
fn recorded_slots(patch: &Path) -> Slots {
    let text = std::fs::read_to_string(patch).expect("saved patch");
    let json: serde_json::Value = serde_json::from_str(&text).expect("patch json");
    json["operations"]
        .as_array()
        .expect("operations")
        .iter()
        .map(|op| {
            (
                op["layer"].as_u64().expect("layer"),
                op["feature"].as_u64().expect("feature"),
            )
        })
        .collect()
}

/// Run `mutation` in a session recording to `patch`, save, and return the
/// slots it recorded.
fn mutated_slots(session: &mut Session, patch: &Path, mutation: &str) -> Slots {
    run(session, &format!("BEGIN PATCH \"{}\";", lql_path(patch)));
    run(session, mutation);
    run(session, "SAVE PATCH;");
    recorded_slots(patch)
}

/// Assert SELECT, DELETE and UPDATE agree on `clause`, each from a fresh
/// session built by `bind`, and return the agreed slot set.
fn assert_same_targets(bind: &dyn Fn() -> (Session, tempfile::TempDir), clause: &str) -> Slots {
    let (mut session, _keep) = bind();
    let listed = listed_slots(&run(
        &mut session,
        &format!("SELECT * FROM EDGES WHERE {clause} LIMIT {ALL_ROWS};"),
    ));

    for mutation in [
        format!("DELETE FROM EDGES WHERE {clause};"),
        format!("UPDATE EDGES SET confidence = 0.5 WHERE {clause};"),
    ] {
        let (mut session, dir) = bind();
        let patch = dir.path().join("targets.vlp");
        let mutated = mutated_slots(&mut session, &patch, &mutation);
        assert_eq!(mutated, listed, "{mutation} vs its SELECT preview");
    }
    listed
}

#[test]
fn select_lists_exactly_the_slots_delete_and_update_target() {
    let slots = |pairs: &[(u64, u64)]| pairs.iter().copied().collect::<Slots>();
    // (clause, the slots it addresses in `LAYERS`)
    let cases: &[(&str, Slots)] = &[
        ("layer = 0", slots(&[(0, 0), (0, 1), (0, 2)])),
        ("feature = 2", slots(&[(0, 2), (1, 2)])),
        ("layer = 0 AND feature = 1", slots(&[(0, 1)])),
        // A slot with no metadata holds no edge and has no score.
        ("layer = 1 AND feature = 1", slots(&[])),
        (r#"entity = "Paris""#, slots(&[(0, 0)])),
        // Substring, case-insensitive: "Paris" and "Spain".
        (r#"entity = "PA""#, slots(&[(0, 0), (1, 2)])),
        // In every slot's top-k, in no slot's top token: addresses nothing.
        (r#"entity = "France""#, slots(&[])),
        ("confidence < 0.8", slots(&[(0, 2), (1, 2)])),
        // Boundaries are exact: 0.88 is `>=` 0.88 and not `>` 0.88.
        ("score >= 0.88", slots(&[(0, 0), (0, 1), (1, 0)])),
        ("score > 0.88", slots(&[(0, 0), (1, 0)])),
        ("score = 0.9", slots(&[(1, 0)])),
        ("score != 0.9", slots(&[(0, 0), (0, 1), (0, 2), (1, 2)])),
        ("layer = 1 AND confidence > 0.8", slots(&[(1, 0)])),
        (
            r#"entity = "Spain" AND score < 0.8 AND layer = 1"#,
            slots(&[(1, 2)]),
        ),
        (r#"entity = "Spain" AND layer = 0"#, slots(&[])),
    ];

    for (clause, expected) in cases {
        let agreed = assert_same_targets(&session_without_a_patch, clause);
        assert_eq!(&agreed, expected, "WHERE {clause}");
    }
}

#[test]
fn a_tombstoned_slot_is_addressed_by_neither_select_nor_a_second_delete() {
    // The starting overlay state is part of the contract: after a DELETE,
    // the slot is gone for SELECT and for the next mutation alike.
    let bind = || {
        let (mut session, dir) = session_without_a_patch();
        let first = dir.path().join("first.vlp");
        run(
            &mut session,
            &format!("BEGIN PATCH \"{}\";", lql_path(&first)),
        );
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
        );
        run(&mut session, "SAVE PATCH;");
        (session, dir)
    };

    let agreed = assert_same_targets(&bind, "layer = 0");
    assert_eq!(agreed, [(0, 1), (0, 2)].into_iter().collect::<Slots>());
}

// ── VINDEX3 ──────────────────────────────────────────────────────────────

fn v3_session() -> (Session, tempfile::TempDir) {
    use larql_vindex::format::vindex3::fixtures::{
        encode_fixture_container, miniature_glimmer, G_VOCAB,
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).expect("checkpoint dir");
    std::fs::create_dir_all(&container).expect("container dir");
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, "where-fixture");
    std::fs::write(
        container.join("tokenizer.json"),
        larql_inference::test_utils::synthetic_tokenizer_json(G_VOCAB),
    )
    .expect("tokenizer");
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(&container)));
    (session, dir)
}

#[test]
fn v3_select_lists_exactly_the_slots_delete_and_update_target() {
    let mut any_matched = false;
    for clause in [
        "layer = 0",
        "feature = 1",
        "layer = 0 AND feature = 0",
        "score > 0.0",
        "score <= 0.0",
        "layer = 1 AND score > 0.0",
        r#"entity = "1""#,
    ] {
        any_matched |= !assert_same_targets(&v3_session, clause).is_empty();
    }
    assert!(
        any_matched,
        "the V3 fixture has edges, so the gate is not vacuous"
    );
}

#[test]
fn v3_mutations_refuse_what_they_cannot_evaluate_and_change_nothing() {
    let (mut session, dir) = v3_session();
    let before = run(
        &mut session,
        &format!("SELECT * FROM EDGES LIMIT {ALL_ROWS};"),
    );
    let patch = dir.path().join("recording.vlp");
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", lql_path(&patch)),
    );

    for (what, clause) in REFUSED_CLAUSES {
        for stmt in [
            format!("DELETE FROM EDGES WHERE {clause};"),
            format!("UPDATE EDGES SET target = \"changed\" WHERE {clause};"),
        ] {
            assert_refused(try_run(&mut session, &stmt), &format!("{what}: {stmt}"));
        }
    }
    let after = run(
        &mut session,
        &format!("SELECT * FROM EDGES LIMIT {ALL_ROWS};"),
    );
    assert_eq!(after, before, "the V3 overlay changed");
    run(&mut session, "SAVE PATCH;");
    assert!(
        recorded_slots(&patch).is_empty(),
        "a refused mutation was recorded"
    );
}

// ── What a mutation refuses that SELECT answers ──────────────────────────

#[test]
fn a_mutation_refuses_entity_with_relation_and_changes_nothing() {
    // SELECT answers that pair with a ranked gate walk over hundreds of
    // slots per layer. A mutation must not inherit it.
    let mut fx = fixture();
    let before = fx.rows();
    let clause = r#"entity = "Paris" AND relation = "capital""#;

    for stmt in [
        format!("DELETE FROM EDGES WHERE {clause};"),
        format!("UPDATE EDGES SET target = \"changed\" WHERE {clause};"),
        format!("DELETE FROM EDGES WHERE {clause} AND layer = 0 AND feature = 0;"),
    ] {
        let err = try_run(&mut fx.session, &stmt).expect_err(&stmt);
        assert!(err.contains(REFUSAL), "{stmt}: {err}");
        assert!(err.contains("`entity` and `relation`"), "{stmt}: {err}");
        assert_eq!(fx.rows(), before, "{stmt}: the overlay changed");
    }
    assert_eq!(fx.recorded_operations(), 0);
}

#[test]
fn select_says_when_its_listing_is_shorter_than_what_matches() {
    let mut fx = fixture();

    let paged = run(&mut fx.session, "SELECT * FROM EDGES LIMIT 2;").join("\n");
    assert!(
        paged.contains(&format!("{EDGE_COUNT} edges match; showing 2")),
        "{paged}"
    );

    let whole = fx.rows().join("\n");
    assert!(!whole.contains("edges match"), "{whole}");
}
