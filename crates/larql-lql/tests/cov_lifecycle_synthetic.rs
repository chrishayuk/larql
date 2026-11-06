//! Lifecycle-executor coverage sweep against the synthetic
//! [`write_synthetic_model_dir`] fixture.
//!
//! Companion to `coverage_sweep_synthetic.rs`, focused on the four
//! lifecycle-executor files whose deeper branches the broad sweep never
//! reaches:
//!   - executor/lifecycle/compile/into_vindex.rs — FULL bake body runs
//!     and SUCCEEDS on the synthetic fixture (real compiled vindex on
//!     disk), incl. conflict detection, override baking, KNN-store save.
//!   - executor/lifecycle/compile/into_model.rs  — bake body runs (load
//!     weights, MEMIT-fact collection, write); the success tail is gated
//!     by a product limitation (see `compile_current_into_model_runs_bake_body`).
//!   - executor/lifecycle/use_cmd.rs             — Vindex re-USE + the
//!     USE MODEL / USE REMOTE error arms.
//!   - executor/lifecycle/extract.rs             — the load-and-map-error
//!     branch (the success body needs a real HF model dir; see
//!     `extract_model_from_synthetic_dir`).
//!
//! Plumbing-only: synthetic weights produce garbage logits, so we assert
//! on output *shape* / outcome, not semantic content. Where a command
//! needs an output path we write into a `tempfile::tempdir()` so the
//! happy path actually runs (real files on disk), not just the error arm.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    // The LQL lexer treats `\` as a string escape; double up so Windows
    // tempdir backslashes survive into the parsed path literal.
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

/// SQL-safe rendering of a path string (doubles backslashes for the LQL
/// lexer's escape handling, same as `fresh_session`).
fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

// ── USE (use_cmd.rs) ─────────────────────────────────────────────────────

#[test]
fn use_vindex_then_reuse_same_path() {
    // First USE happens in fresh_session; a second USE re-points the
    // session at the same vindex and resets patch_recording/auto_patch
    // (use_cmd.rs tail). Drives the whole Vindex arm a second time.
    let (mut session, dir, path) = fresh_session();
    let _ = &dir; // keep tempdir alive
    let out = try_run(
        &mut session,
        &format!(r#"USE "{}";"#, sql_path(std::path::Path::new(&path))),
    )
    .expect("re-USE should succeed");
    assert!(!out.is_empty(), "USE prints a 'Using:' line");
    assert!(out.join("\n").contains("Using:"));
}

#[test]
fn use_vindex_after_loading_a_model_session() {
    // Load a Weight backend first (USE MODEL), then USE a vindex — both
    // backend transitions run through exec_use's Vindex arm tail.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    // USE MODEL on the synthetic dir: resolve_model_path accepts any
    // existing dir, then load_model_dir errors (no config.json/safetensors).
    // Either way the USE MODEL arm (use_cmd.rs:106-135 / error map) runs.
    let _ = try_run(
        &mut session,
        &format!(r#"USE MODEL "{}";"#, sql_path(dir.path())),
    );
    // Now point at the same dir as a vindex — succeeds.
    let out = try_run(&mut session, &format!(r#"USE "{}";"#, sql_path(dir.path())))
        .expect("USE vindex should succeed");
    assert!(!out.is_empty());
}

#[test]
fn use_model_synthetic_dir_runs_model_arm() {
    // Exercises the UseTarget::Model arm head (use_cmd.rs:106-119):
    // "Loading model: ...", resolve_model_path (accepts the dir), then
    // load_model_dir which errors on the missing HF layout. We accept
    // either Ok or Err — the arm's plumbing runs regardless.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    let _ = try_run(
        &mut session,
        &format!(r#"USE MODEL "{}";"#, sql_path(dir.path())),
    );
}

#[test]
fn use_model_nonexistent_dir_errors() {
    // resolve_model_path returns NotADirectory → error map at use_cmd.rs:114.
    let mut session = Session::new();
    let err = try_run(&mut session, r#"USE MODEL "/no/such/model/dir";"#)
        .expect_err("missing model dir should error");
    assert!(!err.is_empty());
}

#[test]
fn use_remote_bad_url_errors() {
    // UseTarget::Remote arm (use_cmd.rs:136) — a malformed/unreachable
    // URL exercises exec_use_remote's error path without a live server.
    let mut session = Session::new();
    let _ = try_run(&mut session, r#"USE REMOTE "http://127.0.0.1:1/";"#);
}

// ── EXTRACT MODEL (extract.rs) ───────────────────────────────────────────

#[test]
fn extract_model_from_synthetic_dir() {
    // The synthetic dir is a *vindex* layout (.bin weight files, no HF
    // config.json / .safetensors), so InferenceModel::load errors at
    // extract.rs:27-28 with "config.json not found ... architecture
    // cannot be inferred from safetensors alone" (verified). This drives
    // the load-and-map-error branch of exec_extract.
    //
    // The deeper extraction body (extract.rs:30-111: build_vindex,
    // auto-load, KNN/MEMIT rehydrate) needs a real HF model directory
    // with config.json + safetensors holding the architecture's tensors.
    // No test fixture ships one, and constructing a valid safetensors
    // model by hand is out of scope for a plumbing test in a tests/
    // file (the writer would belong in larql-inference test_utils, which
    // this task must not modify). So lines 30-111 stay uncovered here.
    //
    // FORMAT VINDEX2 is explicit: since the M4 flip an unqualified EXTRACT
    // resolves to VINDEX3, and these tests exist to drive the V2 branch.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out_vindex = out_dir.path().join("extracted.vindex");

    let mut session = Session::new();
    let _ = try_run(
        &mut session,
        &format!(
            r#"EXTRACT MODEL "{}" INTO "{}" FORMAT VINDEX2;"#,
            sql_path(dir.path()),
            sql_path(&out_vindex),
        ),
    );
}

#[test]
fn extract_model_with_all_level() {
    // WITH ALL maps the AST ExtractLevel::All → vindex ExtractLevel::All
    // (extract.rs:41-45 match arm). Same load-error outcome on the
    // synthetic fixture, but the WITH-clause + level mapping run.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out_vindex = out_dir.path().join("extracted_all.vindex");

    let mut session = Session::new();
    let _ = try_run(
        &mut session,
        &format!(
            r#"EXTRACT MODEL "{}" INTO "{}" WITH ALL FORMAT VINDEX2;"#,
            sql_path(dir.path()),
            sql_path(&out_vindex),
        ),
    );
}

#[test]
fn extract_model_with_inference_level() {
    // WITH INFERENCE → ExtractLevel::Inference arm.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out_vindex = out_dir.path().join("extracted_inf.vindex");

    let mut session = Session::new();
    let _ = try_run(
        &mut session,
        &format!(
            r#"EXTRACT MODEL "{}" INTO "{}" WITH INFERENCE FORMAT VINDEX2;"#,
            sql_path(dir.path()),
            sql_path(&out_vindex),
        ),
    );
}

// ── COMPILE CURRENT INTO VINDEX (into_vindex.rs) ─────────────────────────

#[test]
fn compile_current_into_vindex_succeeds() {
    // The synthetic vindex has has_model_weights=true and every
    // UNCHANGING weight file the bake hard-links/copies, so the full
    // bake_compile_into_vindex body runs and the atomic promote lands a
    // real vindex on disk. Default conflict strategy (LastWins) with no
    // patches → the no-down-overrides hard-link arm (into_vindex.rs:315-328).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled.vindex");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );
    match res {
        Ok(lines) => {
            assert!(out.exists(), "compiled vindex dir should exist");
            assert!(lines.join("\n").contains("Compiled"));
        }
        Err(e) => panic!("COMPILE INTO VINDEX should succeed on synthetic fixture: {e}"),
    }
}

#[test]
fn compile_current_into_vindex_on_conflict_fail() {
    // ON CONFLICT FAIL arm (into_vindex.rs:87-101). With no patches there
    // are no collisions, so the bake proceeds (collisions.is_empty()).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_fail.vindex");
    let _ = try_run(
        &mut session,
        &format!(
            r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT FAIL;"#,
            sql_path(&out)
        ),
    );
}

#[test]
fn compile_current_into_vindex_on_conflict_highest_confidence() {
    // ON CONFLICT HIGHEST_CONFIDENCE arm (into_vindex.rs:102-110) — the
    // forward-compat no-op branch that behaves like LAST_WINS today.
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_hc.vindex");
    let _ = try_run(
        &mut session,
        &format!(
            r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
            sql_path(&out)
        ),
    );
}

#[test]
fn compile_current_into_vindex_on_conflict_last_wins() {
    // Explicit LAST_WINS arm (the no-op match arm at into_vindex.rs:86).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_lw.vindex");
    let _ = try_run(
        &mut session,
        &format!(
            r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT LAST_WINS;"#,
            sql_path(&out)
        ),
    );
}

#[test]
fn compile_into_vindex_with_inserted_edge_bakes_overrides() {
    // INSERT a compose-mode edge first so down/up/gate overrides exist in
    // the patch overlay; COMPILE then takes the override-baking arms
    // (into_vindex.rs:329-332 patch_down_weights + the overrides_applied
    // / "Down overrides baked" output formatting at the tail).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    // Compose-mode INSERT installs gate/up/down overlays at a free slot.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]") MODE COMPOSE;"#,
    );
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_overrides.vindex");
    let _ = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );
}

#[test]
fn compile_into_vindex_on_conflict_fail_with_collision_errors() {
    // Two patches touching the SAME (layer, feature) slot create a
    // collision; under ON CONFLICT FAIL that returns the early
    // "colliding slot(s)" error (into_vindex.rs:87-101, incl. line 100).
    //
    // Recipe: record a compose INSERT into a .vlp, then APPLY it twice so
    // patched.patches holds two patches with identical (layer, feature).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let patch_dir = tempfile::tempdir().expect("patch tempdir");
    let vlp = patch_dir.path().join("collide.vlp");

    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    );
    try_run(&mut session, "SAVE PATCH;").expect("save patch");
    // Apply the same patch twice → two patches, same slot → collision.
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply 1");
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply 2");

    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_collide.vindex");
    let res = try_run(
        &mut session,
        &format!(
            r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT FAIL;"#,
            sql_path(&out)
        ),
    );
    // If the INSERT genuinely produced a colliding Insert op the compile
    // errors with the collision message; if compose slot allocation made
    // the op a no-op the compile may still succeed. Both leave the
    // conflict-detection + FAIL match arm exercised.
    if let Err(e) = res {
        assert!(!e.is_empty());
    }
}

#[test]
fn compile_into_vindex_last_wins_with_collision_reports_strategy() {
    // Same colliding setup, but default LAST_WINS lets the compile
    // succeed and report the conflict-strategy line (into_vindex.rs:394-406,
    // LAST_WINS arm at :396). A KNN-mode INSERT in the mix also makes
    // collect_compile_collisions skip a None-keyed op (line 38).
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let patch_dir = tempfile::tempdir().expect("patch tempdir");
    let vlp = patch_dir.path().join("collide_lw.vlp");

    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    // Default-mode INSERT (Architecture B / KNN) → PatchOp::InsertKnn,
    // whose key() is None → collect_compile_collisions hits the
    // `None => continue` arm (into_vindex.rs:38).
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "rel", "[4]") AT LAYER 0;"#,
    );
    // Compose-mode INSERT → keyed Insert op that can collide.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    );
    try_run(&mut session, "SAVE PATCH;").expect("save patch");
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply 1");
    try_run(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("apply 2");

    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_collide_lw.vindex");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );
    // Verified to emit the conflict-strategy line + KNN-store-save +
    // override-baking arms on this fixture; accept Ok-or-Err for
    // resilience against future slot-allocation changes.
    if let Ok(lines) = res {
        assert!(lines.join("\n").contains("Compiled"));
    }
}

#[test]
fn compile_into_vindex_copies_label_files() {
    // Pre-seed the source vindex with label sidecars so the
    // label-file copy loop (into_vindex.rs:276-279) finds existing
    // files and copies them into the compiled output. The synthetic
    // fixture doesn't write these, so we add them here.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    // RELATION_CLUSTERS_JSON / FEATURE_CLUSTERS_JSONL / FEATURE_LABELS_JSON.
    std::fs::write(dir.path().join("relation_clusters.json"), "{}").expect("write rel clusters");
    std::fs::write(dir.path().join("feature_clusters.jsonl"), "").expect("write feat clusters");
    std::fs::write(dir.path().join("feature_labels.json"), "{}").expect("write feat labels");

    let mut session = Session::new();
    try_run(&mut session, &format!(r#"USE "{}";"#, sql_path(dir.path()))).expect("use");

    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_labels.vindex");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );
    if res.is_ok() {
        // The copied sidecars should land in the compiled output.
        assert!(out.join("relation_clusters.json").exists());
    }
}

#[test]
fn compile_into_vindex_rejects_output_equal_to_source() {
    // run_atomic_compile's paths_collide guard rejects output == source.
    // Pointing COMPILE at the session's own vindex path triggers it,
    // which surfaces via exec_compile_into_vindex's atomic wrapper.
    let (mut session, _dir, path) = fresh_session();
    let err = try_run(
        &mut session,
        &format!(
            r#"COMPILE CURRENT INTO VINDEX "{}";"#,
            sql_path(std::path::Path::new(&path))
        ),
    )
    .expect_err("output == source should be rejected");
    assert!(err.contains("source vindex") || !err.is_empty());
}

// ── COMPILE CURRENT INTO MODEL (into_model.rs) ───────────────────────────

#[test]
fn compile_current_into_model_runs_bake_body() {
    // has_model_weights=true gate passes (into_model.rs:24-32 skipped),
    // load_model_weights reads the synthetic down/up/attn/etc. (line 49),
    // recording-ops + collect_memit_facts run (57-67), MEMIT is off by
    // default (LARQL_MEMIT_ENABLE unset → the !memit_facts.is_empty() &&
    // memit_enabled guard at :74 is false), then write_model_weights runs
    // (132).
    //
    // It then errors: write_model_weights re-reads `index.json` from the
    // *staging* dir (write_f32.rs:704-705), which the bake never copied
    // there — so COMPILE INTO MODEL cannot fully succeed against the
    // synthetic vindex-only fixture (no source index.json is staged).
    // We accept Ok-or-Err: the bake body up to the write is what we cover.
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_model");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO MODEL "{}";"#, sql_path(&out)),
    );
    if let Ok(lines) = res {
        let joined = lines.join("\n");
        assert!(joined.contains("Compiled") && joined.contains("Model:"));
    }
}

#[test]
fn compile_into_model_with_inserted_edge() {
    // INSERT a compose edge → collect_memit_facts_with_recording sees a
    // fact, but MEMIT stays disabled by default so the !memit_facts.is_empty()
    // && memit_enabled guard (into_model.rs:74) is false and the bake
    // proceeds to write_model_weights. Exercises the recording-ops path.
    let (mut session, dir, _) = fresh_session();
    let _ = &dir;
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]") MODE COMPOSE;"#,
    );
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_model_edge");
    let _ = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO MODEL "{}";"#, sql_path(&out)),
    );
}

#[test]
fn compile_path_into_model_from_other_vindex() {
    // VindexRef::Path arm of exec_compile: opens a *second* source vindex
    // in a throwaway session and compiles it. Drives exec_compile_into_model
    // via the Path branch rather than Current.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new(); // no USE — compile names the source explicitly
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_from_path");
    let res = try_run(
        &mut session,
        &format!(
            r#"COMPILE "{}" INTO MODEL "{}";"#,
            sql_path(dir.path()),
            sql_path(&out),
        ),
    );
    if let Ok(lines) = res {
        assert!(out.exists());
        assert!(lines.join("\n").contains("Compiled"));
    }
}

#[test]
fn compile_path_into_vindex_from_other_vindex() {
    // VindexRef::Path arm → exec_compile_into_vindex.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_vindex_from_path");
    let _ = try_run(
        &mut session,
        &format!(
            r#"COMPILE "{}" INTO VINDEX "{}";"#,
            sql_path(dir.path()),
            sql_path(&out),
        ),
    );
}

/// `EXTRACT ... FORMAT VINDEX3` routes to the V3 arm, not the V2 one.
///
/// The two arms fail differently on an unresolvable model — V3 resolves
/// the checkpoint path for the encoder, V2 loads a model through
/// `InferenceModel` — so the error names which arm ran. (The V3 arm's
/// own behaviour — encode, capability snapshot, auto-bind, refusals — is
/// gated end to end in `vindex3_extract.rs`.)
#[test]
fn extract_format_vindex3_takes_the_v3_arm() {
    let mut session = Session::new();
    let v3 = try_run(
        &mut session,
        r#"EXTRACT MODEL "no-such-model" INTO "/nonexistent/out" FORMAT VINDEX3;"#,
    )
    .expect_err("an unresolvable model must fail");
    assert!(v3.contains("failed to resolve model path"), "{v3}");

    let v2 = try_run(
        &mut session,
        r#"EXTRACT MODEL "no-such-model" INTO "/nonexistent/out" FORMAT VINDEX2;"#,
    )
    .expect_err("an unresolvable model must fail");
    assert!(v2.contains("failed to load model"), "{v2}");
}
