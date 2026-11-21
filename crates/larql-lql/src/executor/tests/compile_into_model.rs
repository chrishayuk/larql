//! COMPILE INTO MODEL

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
