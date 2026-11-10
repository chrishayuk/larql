//! **The V2→V3 LQL compatibility gate** — the release criterion for
//! VINDEX3 becoming the default format: the same LQL means the same
//! model, whatever the underlying authority model. VINDEX3 does not
//! replace VINDEX2 until every row below is green.
//!
//! ```text
//! READ        ✓ SELECT   ✓ DESCRIBE   ✓ WALK   ✓ SHOW
//! INFERENCE   ✓ INFER    ✓ GENERATE (V3-only surface, chain-gated)
//! MUTATION    ✓ INSERT KNN   ✓ DELETE   ✓ UPDATE   ✓ MERGE
//!             ✓ INSERT COMPOSE — the full V2 pipeline (capture,
//!               refine, balance, cross-fact, decoys) ported onto the
//!               operand-source seam; staged parity below
//! PATCH       ✓ BEGIN/SAVE/APPLY/REMOVE/SHOW   ✓ stacking order
//! LIFECYCLE   ◐ COMPILE — INTO VINDEX bakes on V3 (equivalence-
//!               gated in vindex3_compile.rs; derived-annotation
//!               refusals documented; INTO MODEL later)
//!             ✓ DIFF — logical-first on V3, gated as the COMPILE
//!               oracle in reverse (meaning ≠ storage); PHYSICAL
//!               subordinate; mixed-generation + INTO PATCH later
//!             ✓ COMPACT INTO VINDEX — semantics-preserving physical
//!               reorganisation, proven by DIFF (SemanticDiff = ∅);
//!               refuses overlay state (COMPILE first); V2's tiered
//!               MINOR/MAJOR remain a later capability on V3
//! ```
//!
//! One source checkpoint (the dense Llama-shaped fixture, LCG-seeded
//! and therefore byte-reproducible) is realised BOTH ways:
//!
//! - **V2**: loaded as `ModelWeights` and extracted with the real
//!   `build_vindex` pipeline (f32 storage, `ExtractLevel::All`);
//! - **V3**: encoded as a VINDEX3 container by `encode_system`.
//!
//! The same LQL script runs against both bindings and must produce
//! equivalent logical results. Preregistered contract for this rung:
//!
//! - **exact**: the feature space — per layer, the set and identity of
//!   `(feature id → top token)`; WALK's per-layer hit feature ids.
//! - **matched by construction**: annotation semantics (`c_score` =
//!   top logit of `embed · feature_down`) — the V3 role derivation
//!   implements the V2 extractor's contract verbatim, and the first
//!   run of this harness is what caught the original divergence
//!   (V3 initially scored against the output head).
//! - **excluded** (explicitly, not silently): relation labels (no
//!   label sidecars exist on either side here) and gate-score display
//!   strings (compared as ids/ordering, not text).
//!
//! Controls precede the parity claim: the extractor of logical rows
//! must be stable across repeated runs of one arm, and must DIFFER
//! across genuinely different models — otherwise "V2 == V3" would be
//! vacuous.

use std::collections::BTreeMap;
use std::path::Path;

/// An UNAMBIGUOUS `[N]` ↔ id N tokenizer: `unk_token` points at the
/// existing `"[0]"` entry instead of aliasing a second surface onto
/// id 0. The shared `synthetic_tokenizer_json` maps both `"[0]"` and
/// `"[UNK]"` to id 0, and the V2 down-meta reader and the V3 view
/// resolve that alias differently — a fixture artifact the first
/// parity run flagged as a false divergence. Parity fixtures must not
/// carry ambiguous vocabularies.
fn unambiguous_tokenizer_json(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, DENSE_LAYERS, DENSE_VOCAB,
    G_VOCAB,
};

/// The one model both realisations are built from, and so the identity both
/// declare and every portable patch names.
///
/// The V2 vindex and the V3 container are the SAME checkpoint. `APPLY PATCH`
/// compares the patch's `base_model` with the bound vindex's declared name,
/// so a patch moves between the two only because they declare one name. They
/// used to declare two spellings of it, which nothing checked.
const PARITY_MODEL: &str = "parity-dense";

/// A model the fixtures are not, for the gates that show a patch does not
/// cross models merely because its slots and widths fit.
const OTHER_MODEL: &str = "parity-dense-other-weights";

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

/// Windows temp paths contain backslashes, which the LQL lexer's escape
/// pass would consume; doubling them leaves the path untouched on every
/// platform.
fn lql_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn session_for(dir: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(dir)));
    session
}

/// The V2 realisation: checkpoint → ModelWeights → real extraction.
fn v2_vindex() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let weights = larql_inference::load_model_dir(checkpoint.path()).expect("load checkpoint");

    let out = tempfile::tempdir().unwrap();
    let tok_json = unambiguous_tokenizer_json(DENSE_VOCAB);
    std::fs::write(out.path().join("tokenizer.json"), &tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        PARITY_MODEL,
        out.path(),
        8,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .expect("build V2 vindex");
    // build_vindex may rewrite dir contents; make sure the tokenizer
    // is present for the V2 loaders.
    std::fs::write(
        out.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    out
}

/// The V3 realisation of the SAME checkpoint (LCG-seeded writer —
/// identical bytes).
fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        PARITY_MODEL,
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    container
}

/// The logical feature space one binding reports: per layer, feature
/// id → top token, read from `SELECT * FROM FEATURES` rows
/// (`L<layer>  F<feat>  <token> …`).
fn feature_space(
    session: &mut Session,
    layers: usize,
    limit: usize,
) -> BTreeMap<(usize, usize), String> {
    let mut space = BTreeMap::new();
    for layer in 0..layers {
        let out = run(
            session,
            &format!("SELECT * FROM FEATURES WHERE layer = {layer} LIMIT {limit};"),
        );
        for line in &out {
            let mut parts = line.split_whitespace();
            let (Some(l), Some(f), Some(token)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let (Some(l), Some(f)) = (l.strip_prefix('L'), f.strip_prefix('F')) else {
                continue;
            };
            let (Ok(l), Ok(f)) = (l.parse::<usize>(), f.parse::<usize>()) else {
                continue;
            };
            space.insert((l, f), token.to_string());
        }
    }
    space
}

/// WALK's logical result: per layer, the hit feature ids in rank order
/// (`  L 0: F14 …`).
fn walk_hits(session: &mut Session, prompt: &str) -> Vec<(usize, usize)> {
    let out = run(session, &format!("WALK \"{prompt}\" TOP 5;"));
    let mut hits = Vec::new();
    for line in &out {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('L') else {
            continue;
        };
        let Some((layer, rest)) = rest.split_once(':') else {
            continue;
        };
        let Ok(layer) = layer.trim().parse::<usize>() else {
            continue;
        };
        let Some(feat) = rest.trim_start().strip_prefix('F') else {
            continue;
        };
        let Some(feat) = feat.split_whitespace().next() else {
            continue;
        };
        let Ok(feat) = feat.parse::<usize>() else {
            continue;
        };
        hits.push((layer, feat));
    }
    hits
}

/// The logical outcome one arm reports after the identical KNN
/// mutation script (V3-LQL-3B): the install layer INSERT chose, and
/// whether the edit is observable through DESCRIBE and through
/// INFER's post-logits override.
fn knn_mutation_outcome(dir: &Path) -> (usize, bool, bool) {
    let mut session = session_for(dir);

    // Pre-screen on this arm: the edge must be absent before the
    // insert, or "present after" proves nothing.
    let before = run(&mut session, r#"DESCRIBE "[2]";"#).join("\n");
    assert!(!before.contains("→ [5]"), "edge pre-exists: {before}");

    let inserted = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
    )
    .join("\n");
    let layer: usize = inserted
        .split(" at L")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no install layer in: {inserted}"));

    let describe = run(&mut session, r#"DESCRIBE "[2]";"#).join("\n");
    let infer = run(&mut session, r#"INFER "The b of [2] is" TOP 3;"#).join("\n");
    let override_leads = infer.contains("knn_override")
        && infer
            .lines()
            .find(|l| l.trim_start().starts_with("1."))
            .is_some_and(|row| row.contains("[5]"));
    (layer, describe.contains("→ [5]"), override_leads)
}

//
// The conceptual rule under test, on both backends:
//
//     VisibleModel = fold(BaseModel, ordered ActivePatches)
//
// Operations never mutate a progressively-corrupted working copy —
// visible state is always derivable from the base plus the ordered
// list of active patches. Removal is therefore recomputation, which
// gives resurrection for free; it is never an inverse operation
// reconstructing destroyed state.

/// Hand-author a `.vlp` carrying `operations`. LQL statements author
/// most patches (`BEGIN PATCH` … `SAVE PATCH;`), but some patch-format
/// operations — `DeleteKnn` today — have no emitting statement yet;
/// the portable artifact is still the contract both backends replay.
fn write_patch(path: &std::path::Path, operations: Vec<larql_vindex::PatchOp>) {
    write_patch_for(path, PARITY_MODEL, operations);
}

/// [`write_patch`] under a stated model identity, so a gate can ask what a
/// backend does with a patch written against some other model.
fn write_patch_for(path: &std::path::Path, model: &str, operations: Vec<larql_vindex::PatchOp>) {
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: model.into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations,
    };
    patch.save(path).expect("write patch");
}

/// Whether the KNN fact `[2] → [5]` is visible through DESCRIBE.
fn knn_fact_visible(session: &mut Session) -> bool {
    run(session, r#"DESCRIBE "[2]";"#)
        .join("\n")
        .contains("→ [5]")
}

/// A word-level tokenizer WITH a whitespace pre-tokenizer, for the
/// compose parity arm: distinct facts must tokenize to distinct
/// canonical prompts, or every capture is the same `[UNK]` residual
/// and refine annihilates the whole constellation on both arms
/// (vacuous parity). Word ids stay inside the dense fixture's vocab.
fn word_tokenizer_json() -> String {
    // One distinct word per canonical decoy prompt (ids 9..18): with a
    // degenerate vocab every decoy tokenizes to the same [UNK] run,
    // giving bitwise-duplicate decoy residuals whose cross-arm noise
    // straddles the Gram-Schmidt near-dependency threshold — the two
    // arms then build different suppress-basis RANKS and refine
    // directions diverge grossly. Real decoy prompts are distinct;
    // the fixture must be too.
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8,"Once":9,"quick":10,"To":11,"Water":12,"long":13,"beginning":14,"weather":15,"She":16,"He":17,"children":18"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

fn v2_vindex_worded() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let weights = larql_inference::load_model_dir(checkpoint.path()).expect("load checkpoint");
    let out = tempfile::tempdir().unwrap();
    let tok_json = word_tokenizer_json();
    std::fs::write(out.path().join("tokenizer.json"), &tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        PARITY_MODEL,
        out.path(),
        8,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .expect("build V2 vindex");
    std::fs::write(out.path().join("tokenizer.json"), word_tokenizer_json()).unwrap();
    out
}

fn v3_container_worded() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        PARITY_MODEL,
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        word_tokenizer_json(),
    )
    .unwrap();
    container
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

mod the_patch_algebra_gates;
mod vindex3_v2_parity_basics;
