//! Executor-core + introspection line-coverage sweep against the
//! synthetic [`write_synthetic_model_dir`] vindex.
//!
//! Companion to `coverage_sweep_synthetic.rs`. Where that file does one
//! representative call per verb, this file drives the *branch* structure
//! of the central statement-dispatch (`executor/mod.rs`), the
//! introspection executors (`executor/introspection.rs`), the backend
//! accessors (`executor/backend.rs`), the compaction executors
//! (`executor/compact.rs`), and the trace executor (`executor/trace.rs`).
//!
//! Plumbing-only: the synthetic weights produce garbage logits, so we
//! assert on output *shape* (Ok-vs-Err, header present, non-empty) and
//! never on semantic model behaviour. Many statements are run twice —
//! once on a loaded vindex (`fresh_session`) and once on a bare
//! `Session::new()` (no `USE`) — so the "no backend" error arms in the
//! accessors (`require_vindex`/`require_patched`/etc.) and dispatch get
//! hit alongside the happy path.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

/// A vindex-backed session pointed at a fresh synthetic fixture.
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

/// Parse + execute, surfacing parse / execute errors as `Err(String)`.
/// Both outcomes are valid for a plumbing sweep — the caller decides.
fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

// ════════════════════════════════════════════════════════════════════
//  executor/backend.rs — accessor "no backend" arms
//
//  Every introspection / mutation / patch verb funnels through one of
//  `require_vindex` / `require_patched` / `require_patched_mut` /
//  `memit_store_mut`. On a bare session those all hit the `_ =>
//  Err(LqlError::NoBackend)` arm (backend.rs:91,119,...). The
//  `relation_classifier()` / `memit_store()` accessors return `None` on
//  a non-Vindex backend (backend.rs:157,177). We exercise both.
// ════════════════════════════════════════════════════════════════════

#[test]
fn no_backend_arms_for_introspection_verbs() {
    let mut session = Session::new();
    // Each of these requires a vindex; on a bare session they must hit
    // the NoBackend accessor arm and return Err.
    for sql in [
        "SHOW RELATIONS;",
        "SHOW LAYERS;",
        "SHOW FEATURES 0;",
        "SHOW ENTITIES;",
        "SHOW COMPACT STATUS;",
        "STATS;",
        "COMPACT MINOR;",
        "COMPACT MAJOR;",
        "SHOW PATCHES;",
        r#"APPLY PATCH "/nonexistent/x.vlp";"#,
        r#"REMOVE PATCH "nope";"#,
    ] {
        let res = try_run(&mut session, sql);
        assert!(res.is_err(), "expected NoBackend error for {sql:?}, got Ok");
    }
}

#[test]
fn mutation_verbs_error_without_backend() {
    let mut session = Session::new();
    // INSERT/DELETE/UPDATE auto-start a patch session, then require a
    // vindex via require_patched_mut → NoBackend error arm.
    for sql in [
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "r", "[2]");"#,
        r#"DELETE FROM EDGES WHERE entity = "[1]";"#,
        r#"UPDATE EDGES SET target = "[2]" WHERE entity = "[1]";"#,
        r#"TRACE "[1]";"#,
    ] {
        let res = try_run(&mut session, sql);
        assert!(res.is_err(), "expected error for {sql:?} without backend");
    }
}

// ════════════════════════════════════════════════════════════════════
//  executor/introspection.rs — SHOW variants and branches
// ════════════════════════════════════════════════════════════════════

#[test]
fn show_relations_mode_and_layer_variants() {
    let (mut session, _dir, _) = fresh_session();
    // Brief (default), VERBOSE, RAW, plus an explicit layer filter and
    // WITH EXAMPLES — each toggles a distinct branch in
    // exec_show_relations (show_raw computation, layer_label, examples).
    for sql in [
        "SHOW RELATIONS;",
        "SHOW RELATIONS VERBOSE;",
        "SHOW RELATIONS RAW;",
        "SHOW RELATIONS BRIEF;",
        "SHOW RELATIONS WITH EXAMPLES;",
        "SHOW RELATIONS AT LAYER 0;",
        "SHOW RELATIONS AT LAYER 1 VERBOSE WITH EXAMPLES;",
    ] {
        let out = try_run(&mut session, sql).expect("SHOW RELATIONS ok");
        // Synthetic features carry no probe labels and no content tokens,
        // so the "(no relations found)" line is the expected shape.
        assert!(!out.is_empty(), "expected output for {sql:?}");
    }
}

#[test]
fn show_layers_range_and_full() {
    let (mut session, _dir, _) = fresh_session();
    // No range → all layers; RANGE keyword form; bare integer-range form;
    // out-of-bounds range → empty body (filtered against loaded_layers).
    for sql in [
        "SHOW LAYERS;",
        "SHOW LAYERS 0-1;",
        "SHOW LAYERS RANGE 0-1;",
        "SHOW LAYERS 5-9;",
    ] {
        let out = try_run(&mut session, sql).expect("SHOW LAYERS ok");
        assert!(!out.is_empty(), "header always present for {sql:?}");
    }
}

#[test]
fn show_features_filters_and_errors() {
    let (mut session, _dir, _) = fresh_session();
    // Plain, with LIMIT, with token/confidence WHERE filters → exercises
    // the token_filter + min_score extraction + per-feature filter loop.
    for sql in [
        "SHOW FEATURES 0;",
        "SHOW FEATURES 0 LIMIT 3;",
        r#"SHOW FEATURES 0 WHERE token = "the";"#,
        r#"SHOW FEATURES 0 WHERE entity = "x" LIMIT 2;"#,
        "SHOW FEATURES 0 WHERE confidence > 0.5;",
        "SHOW FEATURES 1 WHERE score > 2;",
    ] {
        let out = try_run(&mut session, sql).expect("SHOW FEATURES ok");
        assert!(!out.is_empty(), "header always present for {sql:?}");
    }
    // `relation` used to filter on the top token and `c_score` used to
    // be a floor whatever the operator said; neither is a field here.
    for sql in [
        r#"SHOW FEATURES 0 WHERE relation = "x";"#,
        "SHOW FEATURES 1 WHERE c_score > 2;",
    ] {
        let err = try_run(&mut session, sql).expect_err("not a SHOW FEATURES field");
        assert!(err.contains("unsupported WHERE clause"), "{sql}: {err}");
    }
    // A layer with no features → the `nf == 0` early Err branch.
    let err = try_run(&mut session, "SHOW FEATURES 99;").expect_err("no features at layer 99");
    assert!(!err.is_empty());
}

#[test]
fn show_entities_variants() {
    let (mut session, _dir, _) = fresh_session();
    for sql in [
        "SHOW ENTITIES;",
        "SHOW ENTITIES LIMIT 5;",
        "SHOW ENTITIES AT LAYER 0;",
        "SHOW ENTITIES AT LAYER 1 LIMIT 2;",
        "SHOW ENTITIES 0;",
    ] {
        let out = try_run(&mut session, sql).expect("SHOW ENTITIES ok");
        // Synthetic top-tokens are bracketed ids, never named entities,
        // so the "(no entities found)" line fires — still non-empty.
        assert!(!out.is_empty(), "expected output for {sql:?}");
    }
}

#[test]
fn show_models_with_and_without_vindex() {
    // SHOW MODELS scans cwd; runs without a backend.
    let mut bare = Session::new();
    let out = try_run(&mut bare, "SHOW MODELS;").expect("SHOW MODELS ok");
    assert!(!out.is_empty(), "header always present");
    // Also works on a vindex-backed session (same code path).
    let (mut session, _dir, _) = fresh_session();
    let out2 = try_run(&mut session, "SHOW MODELS;").expect("SHOW MODELS ok");
    assert!(!out2.is_empty());
}

#[test]
fn show_compact_status_low_hidden_dim_branch() {
    let (mut session, _dir, _) = fresh_session();
    // hidden_dim of the synthetic model is 16 (< 1024) → the
    // "L2 (MEMIT): not available" branch (introspection.rs:51-56).
    let out = try_run(&mut session, "SHOW COMPACT STATUS;").expect("ok");
    let joined = out.join("\n");
    assert!(joined.contains("Storage engine status"), "got:\n{joined}");
}

// ════════════════════════════════════════════════════════════════════
//  executor/compact.rs — COMPACT MINOR / MAJOR variants & branches
// ════════════════════════════════════════════════════════════════════

#[test]
fn compact_minor_empty_l0_branch() {
    let (mut session, _dir, _) = fresh_session();
    // No INSERTs yet → knn_store empty → the "L0 is empty" early return.
    let out = try_run(&mut session, "COMPACT MINOR;").expect("ok");
    let joined = out.join("\n");
    assert!(joined.contains("L0 is empty"), "got:\n{joined}");
}

#[test]
fn compact_minor_with_populated_l0() {
    let (mut session, _dir, _) = fresh_session();
    // A default (KNN-mode) INSERT populates knn_store, so COMPACT MINOR
    // reaches the promotion loop. The per-entry COMPOSE re-insert may
    // succeed or fail on the tiny synthetic model — both the Ok and Err
    // arms of the loop are valid plumbing outcomes.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let out = try_run(&mut session, "COMPACT MINOR;").expect("ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("COMPACT MINOR") && joined.contains("complete"),
        "expected promotion summary, got:\n{joined}"
    );
}

#[test]
fn compact_major_low_hidden_dim_errors() {
    let (mut session, _dir, _) = fresh_session();
    // hidden_dim 16 < 1024 → COMPACT MAJOR's hidden-dim guard
    // (compact.rs:105-111) errors before any residual capture, for all
    // option forms (plain / FULL / WITH LAMBDA).
    for sql in [
        "COMPACT MAJOR;",
        "COMPACT MAJOR FULL;",
        "COMPACT MAJOR WITH LAMBDA = 0.001;",
        "COMPACT MAJOR FULL WITH LAMBDA = 0.01;",
    ] {
        let err = try_run(&mut session, sql).expect_err("hidden_dim < 1024");
        assert!(
            err.contains("hidden_dim") || err.contains("1024"),
            "expected hidden-dim guard for {sql:?}, got: {err}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════
//  executor/trace.rs — TRACE / EXPLAIN INFER formatting variants
//
//  The synthetic vindex has quant=None and has_model_weights=true, so
//  TRACE proceeds into exec_trace_with_ffn and the formatting branches.
// ════════════════════════════════════════════════════════════════════

#[test]
fn trace_default_summary() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"TRACE "[1]";"#).expect("ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Trace:"),
        "expected trace header, got:\n{joined}"
    );
}

#[test]
fn trace_decompose_variant() {
    let (mut session, _dir, _) = fresh_session();
    // DECOMPOSE → the attn/ffn-norm table branch (trace.rs:222-256).
    let out = try_run(&mut session, r#"TRACE "[1]" DECOMPOSE;"#).expect("ok");
    assert!(out.join("\n").contains("Trace:"));
}

#[test]
fn trace_answer_trajectory_variant() {
    let (mut session, _dir, _) = fresh_session();
    // FOR <token> → the answer-trajectory branch with the who-↑/↓
    // classification ladder (trace.rs:169-219).
    let out = try_run(&mut session, r#"TRACE "[1]" FOR "[2]";"#).expect("ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Answer trajectory") || joined.contains("Trace:"),
        "got:\n{joined}"
    );
}

#[test]
fn trace_layers_filter_variant() {
    let (mut session, _dir, _) = fresh_session();
    // LAYERS range restricts the displayed rows in every formatter.
    for sql in [
        r#"TRACE "[1]" LAYERS 0-1;"#,
        r#"TRACE "[1]" DECOMPOSE LAYERS 0-1;"#,
        r#"TRACE "[1]" FOR "[2]" LAYERS 0-0;"#,
    ] {
        let out = try_run(&mut session, sql).expect("ok");
        assert!(out.join("\n").contains("Trace:"), "got for {sql:?}");
    }
}

#[test]
fn trace_save_requires_positions_all() {
    let (mut session, _dir, _) = fresh_session();
    // SAVE without POSITIONS ALL → the early Err (trace.rs:131-134).
    let err = try_run(
        &mut session,
        r#"TRACE "[1]" SAVE "/tmp/larql_cov_trace.bin";"#,
    )
    .expect_err("SAVE needs POSITIONS ALL");
    assert!(
        err.contains("POSITIONS ALL") || err.contains("ALL"),
        "got: {err}"
    );
}

#[test]
fn trace_save_with_positions_all() {
    let (mut session, _dir, dir_str) = fresh_session();
    // SAVE + POSITIONS ALL → the maybe_save_and_return write path. The
    // TraceWriter may succeed or error on the tiny model; both exercise
    // the save branch. Write into the (temp) vindex dir for cleanup.
    let save_path = format!("{dir_str}/cov_trace.bin").replace('\\', "\\\\");
    let _ = try_run(
        &mut session,
        &format!(r#"TRACE "[1]" POSITIONS ALL SAVE "{save_path}";"#),
    );
}

#[test]
fn trace_positions_last_variant() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"TRACE "[1]" POSITIONS LAST;"#).expect("ok");
}

#[test]
fn trace_with_populated_knn_store() {
    let (mut session, _dir, _) = fresh_session();
    // A KNN INSERT makes knn_store non-empty so TRACE's pending-retrieval
    // override block (trace.rs:106-124) and append_pending_retrieval_override
    // (trace.rs:318-336) are reachable when infer produces an override.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let out = try_run(&mut session, r#"TRACE "[1]";"#).expect("ok");
    assert!(out.join("\n").contains("Trace:"));
}

// ════════════════════════════════════════════════════════════════════
//  executor/mod.rs — patch dispatch arms + patch executors
// ════════════════════════════════════════════════════════════════════

#[test]
fn save_patch_without_session_errors() {
    let (mut session, _dir, _) = fresh_session();
    // No active patch recording → exec_save_patch's ok_or_else Err
    // (mod.rs:366-368).
    let err = try_run(&mut session, "SAVE PATCH;").expect_err("no active patch");
    assert!(err.contains("no active patch") || !err.is_empty());
}

#[test]
fn begin_patch_then_save_roundtrip() {
    let (mut session, _dir, dir_str) = fresh_session();
    // BEGIN PATCH → exec_begin_patch (mod.rs:342-363, non-auto branch).
    let vlp = format!("{dir_str}/cov.vlp").replace('\\', "\\\\");
    let out = try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("begin ok");
    assert!(out.join("\n").contains("Patch session started"));

    // A second BEGIN PATCH while one is active (non-auto) → the
    // "already active" Err (mod.rs:343-346).
    let err = try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#))
        .expect_err("double begin should fail");
    assert!(err.contains("already active") || !err.is_empty());

    // SAVE PATCH with a named session → exec_save_patch happy path
    // (mod.rs:376-407), reading the model name off the Vindex backend.
    let out = try_run(&mut session, "SAVE PATCH;").expect("save ok");
    assert!(out.join("\n").contains("Saved"));
}

#[test]
fn save_patch_anonymous_session_errors() {
    let (mut session, _dir, _) = fresh_session();
    // A bare INSERT auto-starts an anonymous patch (empty path). SAVE
    // PATCH on it → the "anonymous patch session" Err (mod.rs:370-374).
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let err = try_run(&mut session, "SAVE PATCH;").expect_err("anonymous can't save");
    assert!(err.contains("anonymous") || !err.is_empty());
}

#[test]
fn begin_patch_upgrades_auto_patch() {
    let (mut session, _dir, dir_str) = fresh_session();
    // Auto-start an anonymous patch via INSERT, then BEGIN PATCH upgrades
    // it (keeping operations) → mod.rs:351-356 auto-patch branch.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let vlp = format!("{dir_str}/cov2.vlp").replace('\\', "\\\\");
    let out = try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("upgrade ok");
    assert!(out.join("\n").contains("Patch session started"));
    // Now save the upgraded (named) session.
    let out = try_run(&mut session, "SAVE PATCH;").expect("save ok");
    assert!(out.join("\n").contains("Saved"));
}

#[test]
fn apply_patch_missing_file_errors() {
    let (mut session, _dir, _) = fresh_session();
    // exec_apply_patch: path doesn't exist → early Err (mod.rs:412-414).
    let err = try_run(
        &mut session,
        r#"APPLY PATCH "/nonexistent/cov_missing.vlp";"#,
    )
    .expect_err("missing patch file");
    assert!(err.contains("not found") || !err.is_empty());
}

#[test]
fn apply_patch_roundtrip() {
    let (mut session, _dir, dir_str) = fresh_session();
    // Build a real .vlp via BEGIN/INSERT/SAVE, then APPLY it back so the
    // exec_apply_patch happy path (mod.rs:416-433) runs against the
    // PatchedVindex overlay.
    let vlp = format!("{dir_str}/cov_apply.vlp").replace('\\', "\\\\");
    let _ = try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("begin ok");
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let _ = try_run(&mut session, "SAVE PATCH;").expect("save ok");

    // APPLY the patch file we just wrote.
    let apply_sql = format!(r#"APPLY PATCH "{vlp}";"#);
    let res = try_run(&mut session, &apply_sql);
    // Whether the overlay accepts the op or the loader rejects the
    // synthetic-model patch, both drive the apply branch.
    if let Ok(out) = res {
        assert!(out.join("\n").contains("Applied"));
    }
}

#[test]
fn show_patches_variants() {
    let (mut session, _dir, dir_str) = fresh_session();
    // No patches + no overrides → "(no patches applied)" (mod.rs:439-440).
    let out = try_run(&mut session, "SHOW PATCHES;").expect("ok");
    assert!(out.join("\n").contains("no patches applied"));

    // Start a recording + add an op → the "Recording: ... ops pending"
    // tail (mod.rs:471-481) plus the anonymous/overrides body.
    let vlp = format!("{dir_str}/cov_show.vlp").replace('\\', "\\\\");
    let _ = try_run(&mut session, &format!(r#"BEGIN PATCH "{vlp}";"#)).expect("begin ok");
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let out = try_run(&mut session, "SHOW PATCHES;").expect("ok");
    assert!(out.join("\n").contains("Recording"));
}

#[test]
fn show_patches_anonymous_recording_label() {
    let (mut session, _dir, _) = fresh_session();
    // INSERT auto-starts an anonymous recording (empty path) → the
    // "(anonymous)" label branch in exec_show_patches (mod.rs:472-476).
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    let out = try_run(&mut session, "SHOW PATCHES;").expect("ok");
    assert!(out.join("\n").contains("anonymous"));
}

#[test]
fn remove_patch_not_found_errors() {
    let (mut session, _dir, _) = fresh_session();
    // exec_remove_patch with no matching description → Err (mod.rs:502).
    let err =
        try_run(&mut session, r#"REMOVE PATCH "no-such-patch";"#).expect_err("patch not found");
    assert!(err.contains("not found") || !err.is_empty());
}

// ════════════════════════════════════════════════════════════════════
//  executor/mod.rs — remaining dispatch arms (Rebalance, Pipe, Trace
//  dispatch, etc.) exercised so the match in execute() is fully walked.
// ════════════════════════════════════════════════════════════════════

#[test]
fn rebalance_dispatch_arm() {
    let (mut session, _dir, _) = fresh_session();
    // The Rebalance dispatch arm (mod.rs:259-263). Whether the rebalance
    // finds compose installs or errors on the synthetic model, the
    // dispatch + executor entry run.
    for sql in [
        "REBALANCE;",
        "REBALANCE MAX 1;",
        "REBALANCE FLOOR 0.3 CEILING 0.9;",
    ] {
        let _ = try_run(&mut session, sql);
    }
}

#[test]
fn pipe_dispatch_arm() {
    let (mut session, _dir, _) = fresh_session();
    // The Pipe arm (mod.rs:114-118): two statements joined with `|>`.
    // Both sub-statements run and their outputs concatenate.
    let rows = try_run(&mut session, "SHOW MODELS |> SHOW LAYERS;").expect("pipe ok");
    assert!(!rows.is_empty(), "expected concatenated pipe output");
}

#[test]
fn merge_into_self_dispatch() {
    let (mut session, _dir, _) = fresh_session();
    // Merge dispatch arm (mod.rs:254-258) with a missing source → error
    // path, exercising the dispatch + exec_merge entry.
    let err = try_run(&mut session, r#"MERGE "/nonexistent/cov_src.vindex";"#)
        .expect_err("missing source");
    assert!(!err.is_empty());
}

#[test]
fn select_features_and_entities_dispatch() {
    let (mut session, _dir, _) = fresh_session();
    // SELECT FROM FEATURES / ENTITIES dispatch arms (mod.rs:146,147).
    let _ = try_run(&mut session, "SELECT * FROM FEATURES;");
    let _ = try_run(&mut session, "SELECT * FROM ENTITIES;");
    let _ = try_run(&mut session, "SELECT * FROM EDGES LIMIT 2;");
}

#[test]
fn stats_with_explicit_path_arg() {
    let (mut session, _dir, dir_str) = fresh_session();
    // STATS "<path>" → the Some(vindex) branch of the Stats dispatch.
    let p = dir_str.replace('\\', "\\\\");
    let _ = try_run(&mut session, &format!(r#"STATS "{p}";"#));
    // STATS with no arg uses the loaded backend.
    let out = try_run(&mut session, "STATS;").expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn explain_infer_and_walk_dispatch() {
    let (mut session, _dir, _) = fresh_session();
    // Both ExplainMode arms in the Explain dispatch (mod.rs:159-162).
    let out = try_run(&mut session, r#"EXPLAIN INFER "[1]";"#).expect("ok");
    assert!(out.join("\n").contains("Inference trace"));
    let _ = try_run(&mut session, r#"EXPLAIN WALK "[1]";"#).expect("ok");
}
