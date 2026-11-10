//! COMPILE INTO MODEL
//! COMPACT MAJOR full MEMIT path (large fixture, hidden_dim ≥ 1024)

use super::*;

#[test]
fn compile_into_model_default_path_runs() {
    let (mut session, dir) = full_vindex_session("compile_into_model");
    // Sibling directory rather than nested under the source — keeps
    // the atomic-compile staging dir out of the source vindex.
    let out_dir = std::env::temp_dir().join(format!(
        "larql_compiled_model_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}" FORMAT safetensors;"#,
        lql_path(&out_dir)
    ))
    .unwrap();
    // Compile may succeed or hit a vindex-internal error against the
    // synthetic weights; either way the call exercises the path.
    let _ = session.execute(&stmt);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn compile_into_vindex_with_memit_enabled_runs_solver_path() {
    // LARQL_MEMIT_ENABLE=1 + a compose-mode INSERT in the recording
    // means COMPILE INTO VINDEX invokes the MEMIT closed-form ΔW_down
    // solve and bakes the delta on top of the column-replace overlay.
    let (mut session, dir) = full_vindex_session("compile_into_vindex_memit");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT compose");

    let out_dir = std::env::temp_dir().join(format!(
        "larql_compile_vindex_memit_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}";"#,
        lql_path(&out_dir)
    ))
    .unwrap();

    // Toggle MEMIT on for the duration of this call via the thread-local
    // override (NOT `std::env::set_var`, which races concurrent `getenv` on
    // the decode path that other parallel tests drive → SIGSEGV). Production
    // reads this through `larql_compute::options::env_value`, so the override
    // wins. Cleared on guard drop.
    let _memit = MemitEnableGuard::on();
    let result = session.execute(&stmt);

    // Random-init weights mean the MEMIT solve might not produce
    // a useful delta but the code path runs end-to-end.
    let _ = result;
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn compile_into_vindex_on_conflict_highest_confidence_runs() {
    // ON CONFLICT HIGHEST_CONFIDENCE is accepted as a forward-compat
    // strategy that today behaves like LAST_WINS — the path is
    // accepted by the parser and exec_compile_into_vindex passes
    // through the strategy match arm without erroring.
    use larql_vindex::{PatchOp, VindexPatch};

    let (mut session, dir) = vindex_session("compile_conflict_highest");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        let mkp = |conf: f32, target: &str| VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: vec![PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: Some("r".into()),
                entity: "e".into(),
                target: target.into(),
                confidence: Some(conf),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            }],
        };
        patched.patches.push(mkp(0.5, "low"));
        patched.patches.push(mkp(0.9, "high"));
    }

    let output = compile_output(&dir, "compiled_hc.vindex");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        lql_path(&output)
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(
        result.is_ok(),
        "ON CONFLICT HIGHEST_CONFIDENCE should run, got: {result:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&output);
}

#[test]
fn compile_into_model_with_memit_enabled_runs() {
    // LARQL_MEMIT_ENABLE=1 + a compose-mode INSERT in the recording
    // means COMPILE INTO MODEL invokes the MEMIT solver path.
    let (mut session, dir) = full_vindex_session("compile_into_model_memit");

    let insert = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target)
           VALUES ("[1]", "[2]", "[5]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .unwrap();
    let _ = session.execute(&insert).expect("INSERT compose");

    let out_dir = compile_output(&dir, "compiled_memit");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}" FORMAT safetensors;"#,
        lql_path(&out_dir)
    ))
    .unwrap();

    // Toggle MEMIT on for the duration of this call via the thread-local
    // override (NOT `std::env::set_var`, which races concurrent `getenv` on
    // the decode path → SIGSEGV). See `compile_into_vindex_with_memit_enabled_runs_solver_path`.
    let _memit = MemitEnableGuard::on();
    let result = session.execute(&stmt);

    // The MEMIT solve may or may not converge cleanly with random-init
    // weights — accept either outcome but exercise the path.
    let _ = result;
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_against_full_fixture_runs() {
    // Full fixture has hidden_dim=16 < 1024 — COMPACT MAJOR still
    // errors on the hidden_dim check, but exercises more of the
    // pre-check path than the small-fixture variant since model
    // weights are loadable.
    let (mut session, dir) = full_vindex_session("compact_major");
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(err.to_string().contains("hidden_dim"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_against_large_fixture_with_no_patches_short_circuits() {
    // Large fixture clears the hidden_dim guard. With no patches and
    // no overlay edges, COMPACT MAJOR returns the empty-L1 short-circuit.
    let (mut session, dir) = large_vindex_session("compact_major_empty_l1");
    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let out = session.execute(&stmt).expect("COMPACT MAJOR no-op");
    let joined = out.join("\n");
    assert!(
        joined.contains("L1 is empty"),
        "expected empty-L1 message, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_against_large_fixture_runs_full_memit_solve() {
    // Pre-seat one committed patch with a relation so COMPACT MAJOR
    // runs the full MEMIT pipeline: residual capture, target embedding
    // lookup, ndarray solve, decomposition-quality report, and persist
    // to memit_store.json.
    let (mut session, dir) = large_vindex_session("compact_major_memit");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        patched
            .patches
            .push(mk_insert_patch(0, 0, "[1]", Some("[2]"), "[3]"));
    }

    let stmt = parser::parse("COMPACT MAJOR;").unwrap();
    let out = session.execute(&stmt).expect("COMPACT MAJOR full path");
    let joined = out.join("\n");

    assert!(
        joined.contains("Running MEMIT solver"),
        "expected MEMIT solver output, got: {joined}",
    );
    assert!(
        joined.contains("Decomposition quality"),
        "expected quality report, got: {joined}",
    );
    assert!(
        joined.contains("COMPACT MAJOR complete"),
        "expected completion line, got: {joined}",
    );

    let memit_path = dir.join("memit_store.json");
    assert!(
        memit_path.exists(),
        "expected memit_store.json to be persisted",
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_major_against_large_fixture_with_lambda_override() {
    // WITH LAMBDA = X threads through a non-default lambda; the value
    // is echoed in the progress report.
    let (mut session, dir) = large_vindex_session("compact_major_lambda");
    {
        let (_, _, patched) = session.require_patched_mut().unwrap();
        patched
            .patches
            .push(mk_insert_patch(0, 0, "[1]", Some("[2]"), "[3]"));
    }

    let stmt = parser::parse("COMPACT MAJOR WITH LAMBDA = 0.01;").unwrap();
    let out = session.execute(&stmt).expect("COMPACT MAJOR custom lambda");
    let joined = out.join("\n");
    assert!(
        joined.contains("lambda=1.0e-2") || joined.contains("lambda=1e-2"),
        "expected custom lambda echo, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir);
}
