//! Tier-A line-coverage sweep targeting the executor files that sat
//! below the 90% per-file floor and whose remaining red lines need a
//! fixture variant the base synthetic vindex can't express:
//!
//!   * a **no-weights** vindex (`has_model_weights = false`) to drive the
//!     `!use_constellation` / no-weights branches in
//!     `insert/capture.rs`, `insert/knn.rs`, and `insert/mod.rs`.
//!   * an **alphabetic-tokenizer** vindex so the template-decoy loop in
//!     `insert/capture.rs` (which only pushes a decoy for vocab tokens
//!     that decode to alphabetic 3+-char words) actually fires.
//!   * a **custom-layer-bands** vindex so `describe/exec.rs`'s
//!     knowledge / output band formatting branches receive edges (the
//!     "tinymodel" 2-layer fallback collapses every edge into syntax).
//!   * a **feature-labels** sidecar so the relation classifier returns a
//!     non-empty label and `query/infer.rs`'s label-formatting branch
//!     runs.
//!   * a populated KNN store whose stored key matches the TRACE prompt's
//!     residual so the `knn_override` path in `executor/trace.rs` fires.
//!
//! Plumbing-only: synthetic weights produce garbage logits, so every
//! assertion is on output *shape* / Ok-vs-Err — never on semantic model
//! behaviour.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

/// SQL-safe rendering of a path string (doubles backslashes for the LQL
/// lexer's escape handling).
fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// Build a standard synthetic vindex and `USE` it; return the session +
/// the live tempdir (kept alive by the caller) + the path string.
fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    use_dir(dir)
}

/// `USE` an already-populated synthetic dir (after any in-place edits to
/// index.json / sidecars).
fn use_dir(dir: tempfile::TempDir) -> (Session, tempfile::TempDir, String) {
    let mut session = Session::new();
    let parsed = parser::parse(&format!(r#"USE "{}";"#, sql_path(dir.path()))).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    let path_str = dir.path().display().to_string();
    (session, dir, path_str)
}

/// Rewrite the on-disk `index.json` with `has_model_weights = false`.
/// Test-only manipulation of the loader gate — the weight files stay on
/// disk but the loader treats the vindex as browse-only, which is the
/// only way to reach the `!use_constellation` / no-weights INSERT
/// branches through the public Session API.
fn set_no_weights(dir: &std::path::Path) {
    let idx_path = dir.join("index.json");
    let raw = std::fs::read_to_string(&idx_path).expect("read index.json");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse index.json");
    v["has_model_weights"] = serde_json::Value::Bool(false);
    std::fs::write(&idx_path, serde_json::to_string(&v).unwrap()).expect("write index.json");
}

/// Overwrite `index.json`'s `layer_bands` with the given non-overlapping
/// ranges so `resolve_bands` returns them verbatim (instead of the
/// all-(0,1) tinymodel fallback).
fn set_layer_bands(
    dir: &std::path::Path,
    syntax: (usize, usize),
    knowledge: (usize, usize),
    output: (usize, usize),
) {
    let idx_path = dir.join("index.json");
    let raw = std::fs::read_to_string(&idx_path).expect("read index.json");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse index.json");
    v["layer_bands"] = serde_json::json!({
        "syntax": [syntax.0, syntax.1],
        "knowledge": [knowledge.0, knowledge.1],
        "output": [output.0, output.1],
    });
    std::fs::write(&idx_path, serde_json::to_string(&v).unwrap()).expect("write index.json");
}

/// Replace the on-disk `tokenizer.json` with a WordLevel tokenizer whose
/// ids 0..vocab_size decode to alphabetic 3+-char words. The
/// template-decoy loop in `insert/capture.rs` only pushes a decoy for a
/// vocab token whose trimmed decode is all-alphabetic and ≥3 chars — the
/// default `[N]` tokenizer never satisfies that, so its decoy break /
/// push lines stay dead. `[UNK]` maps to id 0 so any out-of-vocab
/// prompt token still hits a valid embedding row.
fn write_alphabetic_tokenizer(dir: &std::path::Path, vocab_size: usize) {
    // A pool of distinct alphabetic words; cycle with a numeric-free
    // suffix scheme that stays alphabetic ("alphaa", "alphab", ...).
    let base = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ];
    let mut vocab = serde_json::Map::new();
    for i in 0..vocab_size {
        let word = if i < base.len() {
            base[i].to_string()
        } else {
            // Alphabetic-only synthetic word for ids past the pool.
            format!("word{}", to_alpha(i))
        };
        vocab.insert(word, serde_json::Value::Number((i as u64).into()));
    }
    vocab.insert("[UNK]".into(), serde_json::Value::Number(0u64.into()));
    let tok = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]" }
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tok).unwrap(),
    )
    .expect("write tokenizer.json");
}

/// Map an integer to an all-lowercase-letters string (base-26, a..z).
fn to_alpha(mut n: usize) -> String {
    let mut s = String::new();
    n += 1;
    while n > 0 {
        n -= 1;
        s.insert(0, (b'a' + (n % 26) as u8) as char);
        n /= 26;
    }
    s
}

/// Build an on-disk vindex whose `down_meta` carries real per-feature
/// `FeatureMeta` (top_token, c_score, top_k). MERGE reads the SOURCE's
/// `feature_meta` to decide whether to write; the standard synthetic
/// fixture (and any vindex compiled via the public API) writes an EMPTY
/// `down_meta.bin`, so MERGE always `continue`s before reaching the
/// conflict-strategy arms. A featured source is the only way to drive
/// `merge.rs`'s loop body (the should_write match incl. the `(None, _)`
/// arm and the three explicit conflict strategies). Mirrors the
/// `make_test_vindex_dir` builder used by the crate's own unit tests.
fn make_featured_source_dir() -> tempfile::TempDir {
    use larql_models::TopKEntry;
    use larql_vindex::ndarray::Array2;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = tempfile::tempdir().expect("tempdir");
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };
    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        // MERGE refuses a source built from another model, so the source
        // names the model the synthetic session is bound to.
        model: "synthetic/tinymodel".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index
        .save_vindex(dir.path(), &mut config)
        .expect("save source vindex");
    std::fs::write(
        dir.path().join("embeddings.bin"),
        vec![0u8; vocab_size * hidden * 4],
    )
    .expect("write embeddings");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
    )
    .expect("write tokenizer");
    dir
}

//  No-weights vindex → browse-only INSERT branches

//  Alphabetic tokenizer → template-decoy loop in capture.rs

//  Custom layer bands → describe/exec.rs knowledge + output bands

//  Feature-labels sidecar → query/infer.rs label-formatting branch

//  Trace KNN-override path (executor/trace.rs)

//  MERGE with a featured source → loop body + conflict arms (merge.rs)

//  COMPACT MINOR promotion-failure branch (executor/compact.rs)

//  Documented-unreachable branches on the synthetic fixture
//
// The following target-file branches cannot be reached through the public
// Session/parser API on the synthetic vindex. They are exercised by
// larql-cli integration tests against real models, or are defensive /
// product-limited guards the synthetic fixture can't trip.
//
//   * executor/mutation/merge.rs:68 — the `(None, _) => true` arm. It
//     fires only when the SOURCE vindex's `feature_meta(layer, feature)`
//     returns Some (so the loop doesn't `continue` at the source-meta
//     check) while the TARGET overlay returns None. The synthetic
//     `make_test_vindex` writes `down_meta = [None; num_layers]`, and the
//     COMPILE path hard-links that empty `down_meta.bin` rather than
//     baking overlay features into it (into_vindex.rs:212-216) — so NO
//     on-disk source built via the public API has a non-None
//     `feature_meta`, and the source-meta check always continues before
//     reaching line 68.
//
//   * executor/compact.rs:114-118, 146, 188-191, 233-234, 277-280, 313,
//     317-321, 327-333 — the entire COMPACT MAJOR MEMIT body. It is gated
//     behind `hidden_dim >= 1024` (compact.rs:105); the synthetic model is
//     16-dim, so COMPACT MAJOR always returns the hidden-dim error at
//     105-111 and never enters the no-weights guard, the MEMIT solve, the
//     reconstruction-warning, or the persist branches. The MAJOR-success
//     body is unreachable without a ≥1024-dim fixture (which
//     `write_synthetic_model_dir` cannot produce).
//
//   * executor/lifecycle/compile/into_vindex.rs:263-264, 325-326 — the
//     `std::fs::copy` fallback taken only when `std::fs::hard_link` fails.
//     Hard-links succeed within a single filesystem; the tempdirs used
//     here (source and output) live on the same mount, so the fallback
//     never trips. Reaching it needs a cross-filesystem output path,
//     which a hermetic test can't reliably arrange.
//
//   * executor/lifecycle/compile/into_vindex.rs:400 — the
//     `CompileConflict::Fail => "FAIL"` strategy-label arm. It lives in
//     the collisions-reporting block (394-407), which only runs when
//     `!collisions.is_empty()`. But under `ON CONFLICT FAIL` a non-empty
//     collision set returns early at 95-99, so control never reaches 394
//     with `on_conflict == Fail` AND collisions present. The arm exists
//     for match exhaustiveness and is unreachable at this call site.
//
//   * executor/lifecycle/compile/into_vindex.rs:322-323 — copy
//     down_weights for MEMIT. Reached only when `memit_results.is_some()`
//     AND `down_overrides.is_empty()`. The MEMIT path requires a recorded
//     compose Insert op, but a compose INSERT also writes a down override
//     to the overlay, so `down_overrides` is never empty when
//     `memit_results` is Some via the public API.
//
//   * executor/query/infer.rs:16-43 — the Backend::Weight (dense, no
//     vindex) INFER arm. It runs only when the session holds a
//     `Backend::Weight`, which is constructed exclusively by `USE MODEL`
//     against a real HuggingFace model directory (config.json +
//     safetensors). The synthetic vindex fixture has no such layout, and
//     `Session.backend` is `pub(crate)` so an integration test cannot set
//     `Backend::Weight` directly. The crate's own unit tests
//     (`src/executor/tests.rs::infer_on_weight_backend_*`) DO cover this
//     arm by constructing the backend in-crate, but llvm-cov accounts the
//     lib-test build and the integration-link build separately, so the
//     integration build's copy of these lines stays uncovered. This caps
//     `query/infer.rs` below 90% line coverage at the per-file summary
//     level on the synthetic fixture: the integration build covers every
//     infer.rs line EXCEPT the Backend::Weight arm (18, 21-42), and the
//     only way to lift it is a real loadable model fixture (the same
//     safetensors-construction work documented as out of scope for the
//     EXTRACT/COMPILE INTO MODEL paths in cov_lifecycle_synthetic.rs).

mod feature_labels_sidecar_query_infer_rs_la;
mod merge_with_a_featured_source_loop_body_c;
mod no_weights_vindex_browse_only_insert_bra;
