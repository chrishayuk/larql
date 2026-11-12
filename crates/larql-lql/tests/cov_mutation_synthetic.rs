//! Mutation-path coverage sweep against the synthetic vindex.
//!
//! Companion to `coverage_sweep_synthetic.rs`. That sweep exercised the
//! *KNN* INSERT path and the no-match UPDATE/DELETE branches; the slots
//! it touched left the COMPOSE-mode pipeline (plan / capture / compose /
//! balance / cross-fact) and the *matching* UPDATE / MERGE / REBALANCE
//! bodies uncovered because:
//!
//!   * a bare `INSERT` defaults to `MODE KNN` (`InsertMode::default()`),
//!     so the compose phase files never ran;
//!   * `UPDATE ... WHERE entity = "[99]"` matches nothing, so the
//!     update body never ran;
//!   * MERGE was only tested against a missing source (early error).
//!
//! This file drives the COMPOSE pipeline end-to-end on the synthetic
//! fixture (which carries model weights, so `use_constellation = true`),
//! then runs UPDATE / MERGE / REBALANCE against the slots a compose
//! INSERT registers.
//!
//! Plumbing-only — synthetic weights produce garbage logits, so we
//! assert on output *shape* / Ok-vs-Err, never on semantic model
//! behaviour. Both outcomes are fine as long as the target branch runs.

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

/// Escape a filesystem path for embedding in an LQL string literal. The lexer
/// treats `\` as an escape (dropping `\U`/`\T`/etc.), so Windows paths must
/// double their backslashes — same fix `fresh_session` applies to the USE path.
fn esc(p: &str) -> String {
    p.replace('\\', "\\\\")
}

// ── COMPOSE-mode INSERT pipeline ───────────────────────────────────────
//
// A single `MODE COMPOSE` INSERT walks plan.rs (default-layer branch
// 86-91), capture.rs (residual + decoy capture 41+, template-decoy loop
// 153-167), compose.rs (install_slots), and balance.rs
// (balance_installed 47-148; cross_fact_regression_check returns early
// on the first insert because installed_edges is empty).

#[test]
fn insert_compose_default_layer_runs_full_pipeline() {
    // No `AT LAYER` → plan.rs takes the default-layer branch (86-91:
    // bands.knowledge.1.saturating_sub(1)). use_constellation is true on
    // the synthetic fixture (has_model_weights), so capture.rs runs the
    // forward pass + decoy capture and balance.rs runs balance_installed.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE COMPOSE;"#,
    )
    .expect("compose insert should succeed on synthetic weights");
    // format_insert_summary always emits at least the "Inserted:" line.
    assert!(!out.is_empty(), "expected a summary line");
    assert!(
        out.iter().any(|l| l.contains("Inserted")),
        "expected an Inserted summary, got: {out:?}"
    );
}

#[test]
fn insert_compose_at_layer_pins_install() {
    // `AT LAYER 0` → plan.rs takes the layer-hint branch (67-77) rather
    // than the default-layer branch. Pins the install to layer 0.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "capital", "[4]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert at layer should succeed");
    assert!(!out.is_empty());
}

#[test]
fn insert_compose_with_alpha_and_confidence() {
    // Exercises the alpha_override + confidence INSERT clauses (mod.rs
    // 63-64) on the compose path, and the alpha-note formatting branch
    // in format_insert_summary (mod.rs 210-211).
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[5]", "capital", "[6]") AT LAYER 0 ALPHA 0.2 CONFIDENCE 0.8 MODE COMPOSE;"#,
    )
    .expect("compose insert with alpha+confidence should succeed");
    assert!(!out.is_empty());
}

#[test]
fn insert_compose_twice_exercises_cross_fact_check() {
    // The FIRST compose insert registers an installed_edge but
    // cross_fact_regression_check returns early (installed_edges empty
    // at check time). The SECOND insert sees a non-empty installed_edges
    // and runs the cross-fact body (balance.rs 189-236: prior-prompt
    // re-infer 199-220, the regressed/converged branches 223/226, and
    // the shrink loop 229-234). Both inserts share the "capital"
    // template so the prior re-infer path is meaningful.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("first compose insert");
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "capital", "[4]") AT LAYER 1 MODE COMPOSE;"#,
    )
    .expect("second compose insert");
    assert!(!out.is_empty());
}

// ── KNN INSERT (knn.rs) ────────────────────────────────────────────────

#[test]
fn insert_knn_explicit_mode_runs_residual_capture() {
    // `MODE KNN` on the synthetic fixture (has_weights = true) takes the
    // forward-pass capture branch (knn.rs 56-89). target_id resolution,
    // prompt build, infer_patched, and the residual-find at install_layer
    // all run. Lines 88-89 (the "no residual captured" error) only fire
    // if the forward pass omits the install layer, which it doesn't here;
    // noted as unreachable on the synthetic fixture below.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert should succeed");
    assert!(out.iter().any(|l| l.contains("KNN store")));
}

#[test]
fn insert_knn_at_layer_hint() {
    // `AT LAYER 1` → knn.rs layer-hint branch (41-42) instead of the
    // default-band branch (44-48).
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "capital", "[4]") AT LAYER 1 MODE KNN;"#,
    )
    .expect("knn insert at layer should succeed");
    assert!(!out.is_empty());
}

// ── UPDATE (update.rs) — matching body ─────────────────────────────────

#[test]
fn update_matching_slot_after_compose_insert() {
    // A compose insert at layer 0 registers feature 0 with metadata in
    // the overlay (compose.rs insert_feature → overrides_meta). UPDATE
    // ... WHERE layer = 0 AND feature = 0 short-circuits resolve_candidates
    // to (0,0) and feature_meta returns Some → the update body runs
    // (update.rs 45-68): the "target"/"top_token" assignment arm (50-53)
    // and update_feature_meta (65-66).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let out = try_run(
        &mut session,
        r#"UPDATE EDGES SET target = "[3]" WHERE layer = 0 AND feature = 0;"#,
    )
    .expect("update matching slot");
    assert!(
        out.iter().any(|l| l.contains("Updated")),
        "expected an Updated summary, got: {out:?}"
    );
}

#[test]
fn update_confidence_field_matching_slot() {
    // Drives the "confidence"/"c_score" assignment arm (update.rs 55-60)
    // against an installed slot.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let out = try_run(
        &mut session,
        r#"UPDATE EDGES SET confidence = 0.5 WHERE layer = 0 AND feature = 0;"#,
    )
    .expect("update confidence");
    assert!(out.iter().any(|l| l.contains("Updated")));
}

#[test]
fn update_unknown_field_is_refused() {
    // An assignment whose field isn't target/top_token/confidence/c_score
    // used to be dropped while the statement answered "Updated". It is
    // refused by name; `tests/update_set_validation.rs` holds the rest.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let err = try_run(
        &mut session,
        r#"UPDATE EDGES SET bogus_field = "x" WHERE layer = 0 AND feature = 0;"#,
    )
    .expect_err("an unknown field is refused");
    assert!(err.contains("bogus_field"), "{err}");
}

#[test]
fn update_no_match_returns_empty_message() {
    // The no-match early return (update.rs 38-40): WHERE that resolves to
    // no candidate. A pinned (layer, feature) with no installed slot still
    // short-circuits to (layer, feature) in resolve_candidates, but
    // feature_meta returns None so the body's `if let Some` is skipped and
    // update_ops stays empty → "Updated 0 features". To hit the genuine
    // empty-matches branch we use a non-pinned entity filter that
    // find_features can't satisfy (synthetic down_meta is all None).
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"UPDATE EDGES SET target = "[3]" WHERE entity = "nope";"#,
    )
    .expect("update no-match");
    assert!(
        out.iter().any(|l| l.contains("no matching features")),
        "expected no-match message, got: {out:?}"
    );
}

// ── MERGE (merge.rs) ───────────────────────────────────────────────────

#[test]
fn merge_valid_source_nonexistent_target_errors() {
    // merge.rs 26-32: target is Some but doesn't exist → the
    // target-not-found error. Source must be a valid vindex to get past
    // the source.exists() check at the top (18-23), so we point source at
    // the live synthetic dir and target at a bogus path.
    let (mut session, _dir, path) = fresh_session();
    let err = try_run(
        &mut session,
        &format!(r#"MERGE "{path}" INTO "/nonexistent/target.vindex";"#),
    )
    .expect_err("missing target should fail");
    assert!(!err.is_empty());
}

#[test]
fn merge_synthetic_into_current_default_strategy() {
    // No INTO clause → merge.rs 35-38 takes the current-backend target.
    // Source = synthetic dir; merging into the (empty) overlay means
    // every source feature_meta is None on the overlay → the
    // `(None, _) => true` match arm (merge.rs 68) fires. Default strategy
    // is KeepSource (41).
    let (mut session, _dir, path) = fresh_session();
    let out = try_run(&mut session, &format!(r#"MERGE "{}";"#, esc(&path)))
        .expect("merge into current should succeed");
    assert!(
        out.iter()
            .any(|l| l.contains("merged") || l.contains("Merged")),
        "expected merge summary, got: {out:?}"
    );
}

#[test]
fn merge_with_keep_source_strategy() {
    // ON CONFLICT KEEP_SOURCE → merge.rs strategy = KeepSource (41 via
    // explicit clause). Conflict arms 69-73 only fire when the overlay
    // already has the feature; the synthetic source carries no down_meta
    // so feature_meta is None for all features and the (None,_) arm
    // dominates. The strategy plumbing (parse → exec_merge) still runs.
    let (mut session, _dir, path) = fresh_session();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{}" ON CONFLICT KEEP_SOURCE;"#, esc(&path)),
    )
    .expect("merge keep_source");
    assert!(!out.is_empty());
}

#[test]
fn merge_with_keep_target_strategy() {
    let (mut session, _dir, path) = fresh_session();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{}" ON CONFLICT KEEP_TARGET;"#, esc(&path)),
    )
    .expect("merge keep_target");
    assert!(!out.is_empty());
}

#[test]
fn merge_with_highest_confidence_strategy() {
    let (mut session, _dir, path) = fresh_session();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#, esc(&path)),
    )
    .expect("merge highest_confidence");
    assert!(!out.is_empty());
}

#[test]
fn merge_explicit_existing_target_returns_path() {
    // merge.rs 33: target Some AND exists → the explicit target_path is
    // returned (not the NoBackend / current-backend fallback). Point both
    // source and target at the same valid synthetic dir.
    let (mut session, _dir, path) = fresh_session();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{p}" INTO "{p}";"#, p = esc(&path)),
    )
    .expect("merge into existing target");
    assert!(out.iter().any(|l| l.contains(&path)));
}

// ── REBALANCE (rebalance.rs) ───────────────────────────────────────────

#[test]
fn rebalance_no_installs_early_return() {
    // installed_edges empty → rebalance.rs 41-46 early return.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, "REBALANCE;").expect("rebalance with no installs");
    assert!(
        out.iter().any(|l| l.contains("no compose-mode installs")),
        "expected the no-installs message, got: {out:?}"
    );
}

#[test]
fn rebalance_after_compose_insert_runs_loop() {
    // A compose insert registers an installed_edge, so REBALANCE runs the
    // fixed-point loop (rebalance.rs 48-152): load weights/tokenizer,
    // infer each fact's canonical prompt (67-91), the scale decision
    // (93-99 — one of the three band branches), the optional down-scale
    // (101-108: 107 any_changed), the converged break (111-112), and the
    // summary band-counting (124-150: in_band 133, below 129, above 131).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let out = try_run(&mut session, "REBALANCE;").expect("rebalance after insert");
    assert!(
        out.iter().any(|l| l.contains("compose installs")),
        "expected rebalance summary, got: {out:?}"
    );
}

#[test]
fn rebalance_with_explicit_floor_ceiling_max() {
    // Drives the floor/ceiling/max clauses (rebalance.rs 37-39) on a
    // populated session so the summary band-classification covers both
    // the "in band" and out-of-band counters under a custom band.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let out = try_run(&mut session, "REBALANCE MAX 2 FLOOR 0.01 CEILING 0.99;")
        .expect("rebalance with bounds");
    assert!(out.iter().any(|l| l.contains("iterations")));
}

// ── Lines NOT reachable through the public Session/parser API on the
//    synthetic fixture (documented per the coverage-floor convention) ──
//
// These branches require fixture properties the synthetic vindex does
// not have. They're covered by larql-cli integration tests against real
// models, or are defensive guards that the success path can't reach.
//
//   * capture.rs 42-45 — the `!use_constellation` early return. The
//     synthetic fixture sets `has_model_weights = true`
//     (test_utils.rs), so plan.rs makes `use_constellation = true` and
//     this browse-only branch never fires through compose INSERT.
//
//   * capture.rs 155, 165-167 — the template-decoy push + the
//     `template_decoys_added >= count` break. The synthetic tokenizer
//     vocab is `[0]`..`[31]`; each id decodes to `"[N]"` whose trimmed
//     form still contains `[`/`]`, so `word.chars().all(is_alphabetic)`
//     (161-164, which IS evaluated/covered) is false for every token.
//     No decoy is ever pushed, so the break and push lines stay dead.
//
//   * knn.rs 88-89 — the "no residual captured at layer" error. The
//     forward pass always emits a residual at the install layer on the
//     synthetic weights, so the `.ok_or_else` never trips.
//
//   * knn.rs 108 — the `?` on the no-weights embedding-key branch.
//     Unreachable because the synthetic fixture has model weights, so
//     KNN takes the residual-capture branch (56-89), never the
//     no-weights branch (90-110).
//
//   * balance.rs 39 — the empty-`installed` early return. exec_insert
//     (mod.rs 90-93) returns an error before calling balance_installed
//     whenever `installed` is empty, so balance is only ever invoked
//     with a non-empty slot list.
