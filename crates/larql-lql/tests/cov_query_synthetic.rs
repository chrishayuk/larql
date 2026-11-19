//! Targeted line-coverage sweep for the query-executor files that sat
//! below the 90% per-file floor: `describe/exec.rs`, `select/entities.rs`,
//! `walk.rs`, `explain.rs`, and `infer.rs`.
//!
//! Companion to `coverage_sweep_synthetic.rs` — that file does a wide
//! shallow pass over every verb; this one drives the specific *success
//! bodies* and *option branches* the wide sweep can't reach because they
//! only fire once the synthetic vindex carries feature metadata. The
//! base [`write_synthetic_model_dir`] fixture has `down_meta = None`, so
//! `feature_meta` returns `None` for every base feature and the walk /
//! infer / entity scans see empty traces. The trick this file uses is to
//! first run `INSERT INTO EDGES … MODE COMPOSE` (which writes heap
//! `down_meta` via `insert_feature`) and `MODE KNN` (which populates the
//! `KnnStore`), so that the downstream verbs have real per-feature
//! metadata to format and classify.
//!
//! Plumbing-only: the synthetic weights produce garbage logits, so every
//! assertion is on output *shape* (header present, row rendered, Ok-vs-Err
//! branch) — never on semantic model behaviour.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    let path_for_sql = dir.path().display().to_string().replace('\\', "\\\\");
    let parsed = parser::parse(&format!(r#"USE "{path_for_sql}";"#)).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    let path_str = dir.path().display().to_string();
    (session, dir, path_str)
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// Helper: run a statement, asserting Ok and returning its rows.
fn ok(session: &mut Session, sql: &str) -> Vec<String> {
    try_run(session, sql).unwrap_or_else(|e| panic!("expected Ok for {sql:?}: {e}"))
}

/// Insert one COMPOSE-mode edge at the given layer with an entity-like
/// target token. COMPOSE writes the feature's heap `down_meta` so later
/// walk / infer / entity scans find a populated `FeatureMeta`.
fn compose_insert(session: &mut Session, entity: &str, relation: &str, target: &str, layer: u32) {
    let sql = format!(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") MODE COMPOSE AT LAYER {layer};"#
    );
    // COMPOSE runs a forward pass + balance loop; on the synthetic vindex
    // this succeeds (garbage logits, but the pipeline runs end to end).
    let _ = try_run(session, &sql).unwrap_or_else(|e| panic!("compose insert failed: {e}"));
}

/// Entity-like (capitalised, all-alphabetic, non-stop-word) target names
/// for the seeded COMPOSE features. The `looks_like_entity` filter in
/// `select/entities.rs` rejects anything containing a digit, so these
/// must be pure letters — a "Name0"-style target with a trailing digit
/// would be silently dropped from the entity scan.
const SEED_TARGETS: &[&str] = &["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];

/// Seed enough COMPOSE features across both synthetic layers that the
/// inference-trace gate-KNN (top-20 against the forward-pass residual)
/// surfaces at least one feature-with-metadata per layer, and the entity
/// scan finds entity-like `top_token`s. A single insert isn't enough —
/// one override gate rarely makes the top-20 — but a dozen high-gate
/// (×30) installs reliably do.
fn seed_compose_features(session: &mut Session) {
    for (i, target) in SEED_TARGETS.iter().enumerate() {
        // One fact per (entity, relation): a key is asserted once
        // (FACT-ID-1), so each layer gets its own relation.
        compose_insert(session, &format!("[{i}]"), &format!("rel{i}a"), target, 0);
        compose_insert(session, &format!("[{i}]"), &format!("rel{i}b"), target, 1);
    }
}

// ── describe/exec.rs ───────────────────────────────────────────────────

#[test]
fn describe_not_found_entity_hits_early_return() {
    // exec.rs:47-48 — `describe_build_query` returns None for an entity
    // that tokenises to nothing in-vocab. The empty string drives the
    // `(not found)` early return.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"DESCRIBE "";"#);
    assert!(
        out.join("\n").contains("(not found)"),
        "expected (not found), got: {out:?}"
    );
}

#[test]
fn describe_no_edges_for_unknown_entity() {
    // exec.rs:75-77 — query vector resolves but no edges/KNN hits, so the
    // `(no edges found)` branch fires.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"DESCRIBE "zzz";"#);
    assert!(
        out.join("\n").contains("(no edges found)"),
        "expected (no edges found), got: {out:?}"
    );
}

#[test]
fn describe_knn_entry_renders_signal_and_syntax_band() {
    // exec.rs:62-71 — KNN-store entries appended to the edge list, then
    // exec.rs:80-106 — signal banner + syntax-band formatting loop. The
    // synthetic family ("tinymodel", 2 layers) has no registered layer
    // bands, so the fallback bands all span (0,1) and every edge lands in
    // the *syntax* bucket (see the unreachable note below for knowledge /
    // output).
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 1;"#,
    )
    .expect("knn insert ok");
    // Default mode renders the row without the `also:` tag suffix; VERBOSE
    // (exercised below) includes the `[knn:capital]` tag. Here we just
    // assert the signal banner + syntax band + target render.
    let out = ok(&mut session, r#"DESCRIBE "[1]";"#);
    let joined = out.join("\n");
    assert!(
        joined.contains("signal:"),
        "expected signal banner: {joined}"
    );
    assert!(
        joined.contains("Syntax (L"),
        "expected syntax band: {joined}"
    );
    assert!(
        joined.contains("[9]"),
        "expected KNN target rendered: {joined}"
    );

    // VERBOSE mode includes the `also: [knn:<relation>]` suffix (the
    // `also` vec built at exec.rs:67).
    let verbose = ok(&mut session, r#"DESCRIBE "[1]" VERBOSE;"#);
    assert!(
        verbose.join("\n").contains("[knn:capital]"),
        "expected knn relation tag in verbose mode: {verbose:?}"
    );
}

#[test]
fn describe_verbose_uses_verbose_max_edges() {
    // exec.rs:95 — the verbose `else` arm of the brief/verbose
    // `max_edges` selection (and exec.rs:106 again via the syntax loop).
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert ok");
    let out = ok(&mut session, r#"DESCRIBE "[1]" VERBOSE;"#);
    assert!(out.join("\n").contains("[9]"), "verbose describe: {out:?}");
}

#[test]
fn describe_brief_and_raw_modes_run() {
    // Exercises the brief `max_edges` arm (exec.rs:92-94) + the raw/brief
    // formatting variants in describe/format.rs.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert ok");
    let _ = ok(&mut session, r#"DESCRIBE "[1]" BRIEF;"#);
    let _ = ok(&mut session, r#"DESCRIBE "[1]" RAW;"#);
}

#[test]
fn describe_relations_only_filter_runs() {
    // RELATIONS ONLY path: with no classifier the KNN edge has no probe
    // label and is filtered out, leaving the `(no edges found)`-shaped
    // empty bands. Exercises the relations_only argument threading.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[9]") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert ok");
    let _ = ok(&mut session, r#"DESCRIBE "[1]" RELATIONS ONLY;"#);
}

// NOTE — describe/exec.rs:108-128 (the `knowledge` and `output` band
// formatting branches) are UNREACHABLE from this fixture. The synthetic
// model's family is "tinymodel" with 2 layers; `LayerBands::for_family`
// returns None for that pair, so `resolve_bands` falls back to bands that
// all span (0,1). `describe_format_and_split` is an if / else-if chain
// that tests `syntax` first, so any edge at layer 0 or 1 (the only layers
// that exist) lands in the syntax bucket and the knowledge/output buckets
// stay empty. Reaching those branches needs a model family with
// *non-overlapping* bands (e.g. real Gemma 3 4B, 34 layers → syntax 0-13,
// knowledge 14-27, output 28-33), which `write_synthetic_model_dir`
// cannot produce without changing the fixture. Covered by larql-cli
// integration tests against real models.

// ── select/entities.rs ─────────────────────────────────────────────────

#[test]
fn select_entities_success_body_renders_rows() {
    // entities.rs:91-108 (scan + aggregate) and :129-131 (row render).
    // COMPOSE inserts write heap `down_meta` whose `top_token` is the
    // literal target string "Name0".."Name5", all of which pass
    // `looks_like_entity`, so the entity scan finds rows.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"SELECT * FROM ENTITIES;"#);
    let joined = out.join("\n");
    assert!(joined.contains("Entity"), "expected header: {joined}");
    assert!(
        SEED_TARGETS.iter().any(|t| joined.contains(t)),
        "expected an entity-like row: {joined}"
    );
}

#[test]
fn select_entities_layer_filter_branch() {
    // entities.rs:65-71 — the `layer` condition match arm (Value::Integer).
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"SELECT * FROM ENTITIES WHERE layer = 1;"#);
    assert!(
        out.join("\n").contains("Entity"),
        "header expected: {out:?}"
    );
}

#[test]
fn select_entities_entity_filter_match_and_miss() {
    // entities.rs:72-78 (entity/token filter arm) + :99-103 (the
    // substring-match continue path). "Nam" matches every Name* token;
    // "Zzz" matches none, exercising the `continue` skip.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);

    // "lph" is a case-insensitive substring of "Alpha".
    let matched = ok(
        &mut session,
        r#"SELECT * FROM ENTITIES WHERE entity = "lph";"#,
    );
    assert!(
        matched.join("\n").contains("Alpha"),
        "filter should match Alpha: {matched:?}"
    );

    let missed = ok(
        &mut session,
        r#"SELECT * FROM ENTITIES WHERE entity = "Zzz";"#,
    );
    assert!(
        missed.join("\n").contains("(no entities found)"),
        "non-matching filter should be empty: {missed:?}"
    );
}

#[test]
fn select_entities_token_field_filter_branch() {
    // entities.rs:73-74 — the `field == "token"` alternative of the
    // entity-filter `find`.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(
        &mut session,
        r#"SELECT * FROM ENTITIES WHERE token = "Alpha";"#,
    );
    assert!(out.join("\n").contains("Alpha"), "token filter: {out:?}");
}

#[test]
fn select_entities_with_limit_truncates() {
    // entities.rs:79 (explicit limit) + :120 truncate.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"SELECT * FROM ENTITIES LIMIT 2;"#);
    assert!(
        out.join("\n").contains("Entity"),
        "header expected: {out:?}"
    );
}

#[test]
fn select_entities_empty_returns_no_entities_message() {
    // entities.rs:133-135 — the `(no entities found)` tail with no inserts.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"SELECT * FROM ENTITIES;"#);
    assert!(
        out.join("\n").contains("(no entities found)"),
        "expected empty message: {out:?}"
    );
}

// ── walk.rs ────────────────────────────────────────────────────────────

#[test]
fn walk_empty_prompt_errors() {
    // walk.rs:29-31 — empty token list early return.
    let (mut session, _dir, _) = fresh_session();
    let err = try_run(&mut session, r#"WALK "";"#).expect_err("empty prompt should error");
    assert!(err.contains("empty"), "expected empty-prompt error: {err}");
}

#[test]
fn walk_mode_pure_string() {
    // walk.rs:55 — the Pure mode label.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"WALK "[1]" MODE PURE;"#);
    assert!(
        out.join("\n").contains("pure (sparse KNN only)"),
        "expected pure mode label: {out:?}"
    );
}

#[test]
fn walk_mode_dense_string() {
    // walk.rs:56 — the Dense mode label.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"WALK "[1]" MODE DENSE;"#);
    assert!(
        out.join("\n").contains("dense (full matmul)"),
        "expected dense mode label: {out:?}"
    );
}

#[test]
fn walk_mode_hybrid_default_string() {
    // walk.rs:57 — the Hybrid / default label.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"WALK "[1]" MODE HYBRID;"#);
    assert!(
        out.join("\n").contains("hybrid (default)"),
        "expected hybrid mode label: {out:?}"
    );
}

#[test]
fn walk_compare_emits_compare_note() {
    // walk.rs:96-100 — the COMPARE note branch (vs the default note).
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"WALK "[1]" COMPARE;"#);
    assert!(
        out.join("\n").contains("COMPARE shows more features"),
        "expected COMPARE note: {out:?}"
    );
}

#[test]
fn walk_default_emits_non_compare_note() {
    // walk.rs:101-104 — the non-compare note + elapsed line at :95.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"WALK "[1]";"#);
    let joined = out.join("\n");
    assert!(joined.contains("ms"), "expected elapsed line: {joined}");
    assert!(
        joined.contains("pure vindex scan (no attention)"),
        "expected default note: {joined}"
    );
}

#[test]
fn walk_layers_range_and_top() {
    // walk.rs:42-48 (range filter) + :17 (TOP). After COMPOSE inserts the
    // per-layer hit-rendering loop (walk.rs:71-93) can also fire if the
    // installed gate makes the embedding-query top-k.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"WALK "[1]" LAYERS 0-1 TOP 5 COMPARE;"#);
    assert!(out.join("\n").contains("Feature scan"), "header: {out:?}");
}

// ── explain.rs ─────────────────────────────────────────────────────────

#[test]
fn explain_walk_verbose_after_compose() {
    // explain.rs:43 (verbose top_k=10), :48 (verbose show_count), :54
    // (verbose down_count=5). Seeding COMPOSE features gives the walk a
    // populated trace so the hit-rendering loop (explain.rs:53-68) can run.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let _ = ok(&mut session, r#"EXPLAIN WALK "[1]" VERBOSE;"#);
}

#[test]
fn explain_walk_non_verbose_after_compose() {
    // explain.rs:43 (non-verbose top_k=5), :50-52 (`.min(5)`), :54
    // (down_count=3).
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let _ = ok(&mut session, r#"EXPLAIN WALK "[1]";"#);
}

#[test]
fn explain_walk_layers_range_after_compose() {
    // explain.rs:35-38 — the LAYERS range filter, with a populated trace.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let _ = ok(&mut session, r#"EXPLAIN WALK "[1]" LAYERS 0-1;"#);
}

#[test]
fn explain_walk_empty_prompt_errors() {
    // explain.rs:26-28 — empty-prompt early return.
    let (mut session, _dir, _) = fresh_session();
    let err = try_run(&mut session, r#"EXPLAIN WALK "";"#).expect_err("empty prompt");
    assert!(err.contains("empty"), "expected empty-prompt error: {err}");
}

// ── infer.rs ───────────────────────────────────────────────────────────

#[test]
fn infer_trace_loop_renders_feature_rows() {
    // infer.rs:108-135 — the "Inference trace" loop. A single COMPOSE
    // install is rarely picked by the top-20 residual gate-KNN, so seed a
    // dozen high-gate features across both layers; that reliably surfaces
    // feature-with-metadata rows, driving the inner loop body (:111-130:
    // empty-label arm at :117, top_token trim at :121, down-token join at
    // :122-129, and the formatted push at :130).
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"INFER "The rel0 of [0] is";"#);
    let joined = out.join("\n");
    assert!(
        joined.contains("Inference trace"),
        "expected trace header: {joined}"
    );
    // After the header, at least one rendered feature row ("  L..").
    let trace_section = &joined[joined.find("Inference trace").unwrap()..];
    let has_row = trace_section
        .lines()
        .skip(1)
        .any(|l| l.trim_start().starts_with('L'));
    assert!(has_row, "expected at least one trace row:\n{trace_section}");
}

#[test]
fn infer_trace_loop_fires_on_multiple_prompts() {
    // Re-exercise infer.rs:108-135 across several prompts to be robust to
    // which features the residual gate-KNN happens to surface.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let mut any_row = false;
    for prompt in ["[0]", "Name0", "[5]", "The rel3 of [3] is"] {
        let out = ok(&mut session, &format!(r#"INFER "{prompt}";"#));
        let joined = out.join("\n");
        let trace_section = &joined[joined.find("Inference trace").unwrap_or(0)..];
        if trace_section
            .lines()
            .skip(1)
            .any(|l| l.trim_start().starts_with('L'))
        {
            any_row = true;
        }
    }
    assert!(
        any_row,
        "expected at least one prompt to surface a trace row"
    );
}

#[test]
fn infer_compare_emits_dense_section() {
    // infer.rs:137-148 — the COMPARE dense-prediction tail.
    let (mut session, _dir, _) = fresh_session();
    seed_compose_features(&mut session);
    let out = ok(&mut session, r#"INFER "[0]" COMPARE;"#);
    assert!(
        out.join("\n").contains("Predictions (dense)"),
        "expected dense section: {out:?}"
    );
}

#[test]
fn infer_top_k_and_predictions_render() {
    // infer.rs:81-97 — walk-FFN prediction rendering with an explicit TOP.
    let (mut session, _dir, _) = fresh_session();
    let out = ok(&mut session, r#"INFER "[1]" TOP 3;"#);
    assert!(
        out.join("\n").contains("Predictions (walk FFN)"),
        "expected predictions header: {out:?}"
    );
}
