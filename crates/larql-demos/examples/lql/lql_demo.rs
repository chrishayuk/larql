//! LQL End-to-End Demo — parse, session, execute, error handling
//!
//! Demonstrates the full LQL flow: parse statements, manage session state,
//! execute against the (absent) backend, and handle errors gracefully.
//!
//! Run: cargo run -p larql-demos --example lql_demo

use larql_lql::{parse, run_batch, Session};

fn main() {
    println!("=== LQL End-to-End Demo (spec v0.4) ===\n");

    // ── Session lifecycle ──
    section("Session Lifecycle");

    let mut session = Session::new();

    // Before USE: every query should fail with NoBackend
    demonstrate(&mut session, "STATS;", "STATS without backend");
    demonstrate(
        &mut session,
        r#"WALK "test" TOP 5;"#,
        "WALK without backend",
    );
    demonstrate(
        &mut session,
        r#"DESCRIBE "France";"#,
        "DESCRIBE without backend",
    );
    demonstrate(
        &mut session,
        "SELECT * FROM EDGES;",
        "SELECT without backend",
    );
    demonstrate(
        &mut session,
        r#"EXPLAIN WALK "test";"#,
        "EXPLAIN without backend",
    );
    demonstrate(
        &mut session,
        "SHOW RELATIONS;",
        "SHOW RELATIONS without backend",
    );
    demonstrate(&mut session, "SHOW LAYERS;", "SHOW LAYERS without backend");
    demonstrate(
        &mut session,
        "SHOW FEATURES 26;",
        "SHOW FEATURES without backend",
    );

    // SHOW MODELS works without a backend (scans CWD)
    demonstrate(&mut session, "SHOW MODELS;", "SHOW MODELS (always works)");

    // ── USE errors ──
    section("USE Errors");

    demonstrate(
        &mut session,
        r#"USE "/nonexistent/fake.vindex";"#,
        "USE nonexistent vindex",
    );

    demonstrate(
        &mut session,
        r#"USE MODEL "/nonexistent/model";"#,
        "USE MODEL (nonexistent — shows error)",
    );

    demonstrate(
        &mut session,
        r#"USE MODEL "/nonexistent/model" AUTO_EXTRACT;"#,
        "USE MODEL AUTO_EXTRACT (nonexistent — shows error)",
    );

    // ── Weight backend: demonstrate which ops work, which need vindex ──
    section("Weight Backend (USE MODEL)");

    // We can't actually load a model in demo, but show that vindex-only ops
    // give clear errors when no backend is loaded (similar to Weight backend).
    // The Weight backend tests cover the actual behavior.
    println!("  (Skipped: requires a real model on disk.)");
    println!("  When USE MODEL succeeds:");
    println!("    INFER, EXPLAIN INFER, STATS → work (dense inference)");
    println!("    WALK, DESCRIBE, SELECT, INSERT → error with EXTRACT suggestion");
    println!("  See: executor::tests::weight_backend_* tests");

    // ── Operations requiring backend ──
    section("Operations Without Backend");

    demonstrate(
        &mut session,
        r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#,
        "COMPILE (requires backend)",
    );

    demonstrate(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
        "INSERT (requires backend)",
    );

    demonstrate(
        &mut session,
        r#"DELETE FROM EDGES WHERE entity = "x";"#,
        "DELETE (requires backend)",
    );

    demonstrate(
        &mut session,
        r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#,
        "UPDATE (requires backend)",
    );

    // ── Batch execution ──
    section("Batch Execution");

    let batch = r#"
-- This is a batch of LQL statements
SHOW MODELS;
-- This should produce an error (no backend)
STATS;
-- Another that works
SHOW MODELS;
"#;

    println!("  Input batch:");
    for line in batch.lines() {
        if !line.trim().is_empty() {
            println!("    {}", line.trim());
        }
    }
    println!();

    match run_batch(batch) {
        Ok(lines) => {
            println!("  Output ({} lines):", lines.len());
            for line in &lines {
                println!("    {line}");
            }
        }
        Err(e) => println!("  Batch error: {e}"),
    }

    // ── Parse error handling ──
    section("Parse Error Handling");

    let bad_inputs = vec![
        ("Empty", ""),
        ("Unknown keyword", "FOOBAR;"),
        ("Missing prompt", "WALK TOP 5;"),
        ("Missing FROM", r#"SELECT * WHERE entity = "x";"#),
        (
            "Missing VALUES",
            "INSERT INTO EDGES (entity, relation, target);",
        ),
        ("Bad SHOW noun", "SHOW FOOBAR;"),
        ("Unterminated string", r#"WALK "unterminated"#),
    ];

    for (label, input) in &bad_inputs {
        match parse(input) {
            Ok(stmt) => println!("  {label}: unexpected success → {:?}", stmt),
            Err(e) => println!("  {label}: {e}"),
        }
    }

    // ── Spec compliance: all statement types ──
    section("Spec Compliance: All Statement Types (v0.4)");

    let all_statements = vec![
        // Lifecycle
        (
            "EXTRACT",
            r#"EXTRACT MODEL "m" INTO "o" COMPONENTS FFN_GATE, FFN_DOWN LAYERS 0-33;"#,
        ),
        (
            "EXTRACT (inference)",
            r#"EXTRACT MODEL "m" INTO "o" WITH INFERENCE;"#,
        ),
        ("EXTRACT (all)", r#"EXTRACT MODEL "m" INTO "o" WITH ALL;"#),
        (
            "COMPILE",
            r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#,
        ),
        (
            "COMPILE INTO VINDEX",
            r#"COMPILE CURRENT INTO VINDEX "baked.vindex";"#,
        ),
        (
            "COMPILE INTO VINDEX ON CONFLICT LAST_WINS",
            r#"COMPILE CURRENT INTO VINDEX "baked.vindex" ON CONFLICT LAST_WINS;"#,
        ),
        (
            "COMPILE INTO VINDEX ON CONFLICT HIGHEST_CONFIDENCE",
            r#"COMPILE CURRENT INTO VINDEX "baked.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        ),
        (
            "COMPILE INTO VINDEX ON CONFLICT FAIL",
            r#"COMPILE CURRENT INTO VINDEX "baked.vindex" ON CONFLICT FAIL;"#,
        ),
        ("DIFF", r#"DIFF "a.vindex" CURRENT;"#),
        (
            "DIFF (relation)",
            r#"DIFF "a.vindex" "b.vindex" RELATION "capital" LIMIT 20;"#,
        ),
        ("USE (vindex)", r#"USE "path.vindex";"#),
        (
            "USE MODEL",
            r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#,
        ),
        (
            "USE REMOTE",
            r#"USE REMOTE "https://models.example.com/larql";"#,
        ),
        // Query
        (
            "WALK",
            r#"WALK "prompt" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#,
        ),
        (
            "SELECT",
            r#"SELECT entity, target FROM EDGES WHERE relation = "capital" ORDER BY confidence DESC LIMIT 10;"#,
        ),
        (
            "SELECT NEAREST",
            r#"SELECT * FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
        ),
        // DESCRIBE bands
        ("DESCRIBE", r#"DESCRIBE "France";"#),
        ("DESCRIBE SYNTAX", r#"DESCRIBE "def" SYNTAX;"#),
        ("DESCRIBE KNOWLEDGE", r#"DESCRIBE "France" KNOWLEDGE;"#),
        ("DESCRIBE OUTPUT", r#"DESCRIBE "France" OUTPUT;"#),
        ("DESCRIBE ALL", r#"DESCRIBE "France" ALL LAYERS;"#),
        (
            "DESCRIBE AT LAYER",
            r#"DESCRIBE "France" AT LAYER 26 RELATIONS ONLY;"#,
        ),
        // EXPLAIN
        (
            "EXPLAIN WALK",
            r#"EXPLAIN WALK "prompt" LAYERS 24-33 VERBOSE;"#,
        ),
        ("EXPLAIN INFER", r#"EXPLAIN INFER "prompt" TOP 5;"#),
        // Inference
        ("INFER", r#"INFER "prompt" TOP 5 COMPARE;"#),
        // Mutation
        (
            "INSERT",
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c") AT LAYER 26 CONFIDENCE 0.8;"#,
        ),
        (
            "INSERT COMPOSE ALPHA",
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") ALPHA 0.5 MODE COMPOSE;"#,
        ),
        (
            "INSERT all COMPOSE knobs",
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "r", "b") AT LAYER 24 CONFIDENCE 0.9 ALPHA 0.3 MODE COMPOSE;"#,
        ),
        (
            "DELETE",
            r#"DELETE FROM EDGES WHERE entity = "x" AND layer = 26;"#,
        ),
        (
            "DELETE by slot",
            r#"DELETE FROM EDGES WHERE layer = 26 AND feature = 8821;"#,
        ),
        (
            "UPDATE",
            r#"UPDATE EDGES SET target = "y", confidence = 0.9 WHERE entity = "x";"#,
        ),
        (
            "UPDATE by slot",
            r#"UPDATE EDGES SET target = "London" WHERE layer = 26 AND feature = 8821;"#,
        ),
        (
            "MERGE",
            r#"MERGE "src.vindex" INTO "dst.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#,
        ),
        ("REBALANCE", "REBALANCE;"),
        (
            "REBALANCE (full)",
            "REBALANCE UNTIL CONVERGED MAX 16 FLOOR 0.3 CEILING 0.9;",
        ),
        // Patches
        ("BEGIN PATCH", r#"BEGIN PATCH "test.vlp";"#),
        ("SAVE PATCH", "SAVE PATCH;"),
        ("APPLY PATCH", r#"APPLY PATCH "test.vlp";"#),
        ("SHOW PATCHES", "SHOW PATCHES;"),
        ("REMOVE PATCH", r#"REMOVE PATCH "test.vlp";"#),
        // Introspection
        (
            "SHOW RELATIONS",
            "SHOW RELATIONS AT LAYER 26 WITH EXAMPLES;",
        ),
        ("SHOW RELATIONS VERBOSE", "SHOW RELATIONS VERBOSE;"),
        ("SHOW RELATIONS RAW", "SHOW RELATIONS RAW;"),
        ("SHOW LAYERS", "SHOW LAYERS RANGE 0-10;"),
        (
            "SHOW FEATURES",
            r#"SHOW FEATURES 26 WHERE token = "Paris" LIMIT 5;"#,
        ),
        ("SHOW ENTITIES", "SHOW ENTITIES LIMIT 50;"),
        (
            "SHOW ENTITIES AT LAYER",
            "SHOW ENTITIES AT LAYER 26 LIMIT 20;",
        ),
        ("SHOW MODELS", "SHOW MODELS;"),
        ("STATS", r#"STATS "path.vindex";"#),
        ("SHOW COMPACT STATUS", "SHOW COMPACT STATUS;"),
        ("COMPACT MINOR", "COMPACT MINOR;"),
        ("COMPACT MAJOR", "COMPACT MAJOR;"),
        ("COMPACT MAJOR FULL", "COMPACT MAJOR FULL;"),
        (
            "COMPACT MAJOR WITH LAMBDA",
            "COMPACT MAJOR WITH LAMBDA = 0.001;",
        ),
        // EXPLAIN INFER WITH ATTENTION
        (
            "EXPLAIN INFER WITH ATTENTION",
            r#"EXPLAIN INFER "prompt" TOP 5 WITH ATTENTION;"#,
        ),
        // TRACE
        ("TRACE", r#"TRACE "The capital of France is";"#),
        (
            "TRACE FOR",
            r#"TRACE "The capital of France is" FOR "Paris";"#,
        ),
        (
            "TRACE DECOMPOSE LAYERS",
            r#"TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;"#,
        ),
        (
            "TRACE POSITIONS ALL SAVE",
            r#"TRACE "The capital of France is" POSITIONS ALL SAVE "out.trace";"#,
        ),
        (
            "TRACE full",
            r#"TRACE "The capital of France is" FOR "Paris" DECOMPOSE LAYERS 20-30 POSITIONS LAST SAVE "out.trace";"#,
        ),
        // Pipe
        ("PIPE", r#"WALK "test" TOP 5 |> EXPLAIN WALK "test";"#),
    ];

    let mut ok = 0;
    let mut fail = 0;
    for (label, input) in &all_statements {
        match parse(input) {
            Ok(_) => {
                println!("  {:<20} OK", label);
                ok += 1;
            }
            Err(e) => {
                println!("  {:<20} FAIL — {}", label, e);
                fail += 1;
            }
        }
    }

    println!("\n  Result: {ok}/{} passed", ok + fail);
    if fail > 0 {
        std::process::exit(1);
    }

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("\n── {} ──\n", name);
}

fn demonstrate(session: &mut Session, input: &str, label: &str) {
    let stmt = match parse(input) {
        Ok(s) => s,
        Err(e) => {
            println!("  {label}: parse error — {e}");
            return;
        }
    };

    match session.execute(&stmt) {
        Ok(lines) => {
            println!("  {label}: OK");
            for line in lines.iter().take(3) {
                println!("    {line}");
            }
            if lines.len() > 3 {
                println!("    ... ({} more lines)", lines.len() - 3);
            }
        }
        Err(e) => {
            println!("  {label}: {e}");
        }
    }
}
