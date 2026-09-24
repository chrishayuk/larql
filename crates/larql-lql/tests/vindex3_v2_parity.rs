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
        "parity/dense",
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
        "parity-dense",
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

/// Control 1: the instrument is stable — one arm, twice, identical.
#[test]
fn the_parity_instrument_is_stable_across_runs() {
    let v3 = v3_container();
    let mut a = session_for(v3.path());
    let mut b = session_for(v3.path());
    assert_eq!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64)
    );
    assert_eq!(walk_hits(&mut a, "[3]"), walk_hits(&mut b, "[3]"));
}

/// Control 2: the instrument detects genuinely different models —
/// the dense fixture's feature space is not the miniature's.
#[test]
fn the_parity_instrument_detects_different_models() {
    let dense = v3_container();
    let mini_checkpoint = tempfile::tempdir().unwrap();
    let mini = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        mini_checkpoint.path(),
        mini.path(),
        "parity-mini",
    );
    std::fs::write(
        mini.path().join("tokenizer.json"),
        unambiguous_tokenizer_json(G_VOCAB),
    )
    .unwrap();

    let mut a = session_for(dense.path());
    let mut b = session_for(mini.path());
    assert_ne!(
        feature_space(&mut a, DENSE_LAYERS, 64),
        feature_space(&mut b, DENSE_LAYERS, 64),
        "instrument cannot tell different models apart"
    );
}

/// THE gate: one checkpoint, two formats, one script — the same
/// logical feature space and the same walk results.
#[test]
fn v2_and_v3_report_the_same_logical_results() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut v2_session = session_for(v2.path());
    let mut v3_session = session_for(v3.path());

    // Feature space: identity and annotation, exact.
    let v2_space = feature_space(&mut v2_session, DENSE_LAYERS, 300);
    let v3_space = feature_space(&mut v3_session, DENSE_LAYERS, 300);
    assert!(!v2_space.is_empty(), "V2 arm reported no features");
    assert_eq!(
        v2_space, v3_space,
        "the two formats disagree about the feature space"
    );

    // WALK: same prompt, same per-layer hit ids in the same order.
    let v2_hits = walk_hits(&mut v2_session, "[3]");
    let v3_hits = walk_hits(&mut v3_session, "[3]");
    assert!(!v2_hits.is_empty(), "V2 arm walked to nothing");
    assert_eq!(v2_hits, v3_hits, "walk results diverge between formats");

    // DESCRIBE runs on both and agrees about edge presence.
    let v2_describe = run(&mut v2_session, r#"DESCRIBE "[3]";"#).join("\n");
    let v3_describe = run(&mut v3_session, r#"DESCRIBE "[3]";"#).join("\n");
    assert_eq!(
        v2_describe.contains("(no edges found)"),
        v3_describe.contains("(no edges found)"),
        "DESCRIBE disagrees about edge presence:\nV2: {v2_describe}\nV3: {v3_describe}"
    );
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

/// The mutation half of the parity claim: the same INSERT lands on the
/// same layer and is observable through the same statements on both
/// formats — and the outcome is affirmative, not vacuously equal.
#[test]
fn v2_and_v3_agree_after_identical_knn_mutations() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let v2_outcome = knn_mutation_outcome(v2.path());
    let v3_outcome = knn_mutation_outcome(v3.path());
    assert_eq!(
        v2_outcome, v3_outcome,
        "the two formats disagree about the mutation's observable outcome"
    );
    let (_, describe_shows, override_leads) = v2_outcome;
    assert!(describe_shows, "the edit must be visible to DESCRIBE");
    assert!(override_leads, "the stored target must override INFER");
}

/// The feature-mutation half of the parity claim (3B rung 2): the
/// identical UPDATE + DELETE script leaves both formats reporting the
/// same logical feature space — and genuinely changed it (affirmative
/// control against a vacuous pass).
#[test]
fn v2_and_v3_agree_after_identical_feature_mutations() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut v2_session = session_for(v2.path());
    let mut v3_session = session_for(v3.path());

    let pristine = feature_space(&mut v2_session, DENSE_LAYERS, 300);

    for session in [&mut v2_session, &mut v3_session] {
        run(
            session,
            r#"UPDATE EDGES SET target = "[7]", confidence = 0.5 WHERE layer = 1 AND feature = 1;"#,
        );
        run(
            session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 2;",
        );
        // V2's statement contract: an UPDATE on the tombstoned slot
        // matches nothing on either backend.
        let out = run(
            session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 0 AND feature = 2;"#,
        )
        .join("\n");
        assert!(out.contains("no matching features"), "{out}");
    }

    let v2_space = feature_space(&mut v2_session, DENSE_LAYERS, 300);
    let v3_space = feature_space(&mut v3_session, DENSE_LAYERS, 300);
    assert_eq!(
        v2_space, v3_space,
        "the two formats disagree about the mutated feature space"
    );
    assert_ne!(v2_space, pristine, "the script must have changed the space");
    assert_eq!(v2_space.get(&(1, 1)).map(String::as_str), Some("[7]"));
    assert!(
        !v2_space.contains_key(&(0, 2)),
        "the delete must be visible"
    );

    // WALK agrees about the mutated space too (the tombstone filter
    // runs inside each backend's own scan path).
    let v2_hits = walk_hits(&mut v2_session, "[3]");
    let v3_hits = walk_hits(&mut v3_session, "[3]");
    assert_eq!(v2_hits, v3_hits, "walks diverge after mutation");
}

/// MERGE of one V2 source lands identically on a V2 target and a V3
/// target: same merged/skipped counts, same resulting feature space
/// (the V3 overlay then holds every slot as an override — reading
/// through it must equal V2's overlay reads).
#[test]
fn merge_of_a_v2_source_lands_on_both_backends() {
    let source = v2_vindex();
    let v2_target = v2_vindex();
    let v3_target = v3_container();

    let mut outcomes = Vec::new();
    for target in [v2_target.path(), v3_target.path()] {
        let mut session = session_for(target);
        let out = run(
            &mut session,
            &format!("MERGE \"{}\";", lql_path(source.path())),
        )
        .join("\n");
        let counts = out
            .lines()
            .find(|l| l.contains("features merged"))
            .unwrap_or_else(|| panic!("no merge report in {out}"))
            .trim()
            .to_string();
        outcomes.push((counts, feature_space(&mut session, DENSE_LAYERS, 300)));
    }

    assert_eq!(outcomes[0].0, outcomes[1].0, "merge counts diverge");
    assert!(
        outcomes[0].0.starts_with(char::is_numeric) && !outcomes[0].0.starts_with('0'),
        "the merge must have written features: {}",
        outcomes[0].0
    );
    assert_eq!(
        outcomes[0].1, outcomes[1].1,
        "post-merge feature spaces diverge"
    );

    // The conflict strategy's losing arm, on the V3 target: KEEP_TARGET
    // skips every already-present slot.
    let mut session = session_for(v3_target.path());
    run(
        &mut session,
        &format!("MERGE \"{}\";", lql_path(source.path())),
    );
    let out = run(
        &mut session,
        &format!(
            "MERGE \"{}\" ON CONFLICT KEEP_TARGET;",
            lql_path(source.path())
        ),
    )
    .join("\n");
    assert!(out.contains("0 features merged"), "{out}");
}

/// MERGE into a tokenizerless V3 binding refuses naming the missing
/// browse capability — after the source loaded, before any write.
#[test]
fn merge_refuses_on_a_tokenizerless_v3_target() {
    let source = v2_vindex();
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "tokless-merge",
    );
    let mut session = session_for(container.path());
    let stmt = format!("MERGE \"{}\";", lql_path(source.path()));
    let parsed = parse(&stmt).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("no tokenizer, no browse view, no merge");
    assert!(err.to_string().contains("tokenizer.json"), "{err}");
}

/// The stacking invariant, on both backends: **overlay operations are
/// logical facts; replay order determines visible state**. Two patches
/// write the same slot — the later application wins; removing one
/// replays the remainder; applying them in the opposite order flips
/// the outcome. All four states must agree across formats.
#[test]
fn patch_stacking_replays_in_order_on_both_backends() {
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_a = lql_path(&patch_dir.path().join("a.vlp"));
    let patch_b = lql_path(&patch_dir.path().join("b.vlp"));

    // Author the two patches once, from a V2 session (the .vlp is the
    // portable artifact; both backends must replay it identically).
    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_a}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{patch_b}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[8]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    // One arm's observable state: (slot (1,1) token, slot (0,1) alive?).
    let state = |session: &mut Session| -> (Option<String>, bool) {
        let space = feature_space(session, DENSE_LAYERS, 300);
        (space.get(&(1, 1)).cloned(), space.contains_key(&(0, 1)))
    };

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut outcomes = Vec::new();
    for target in [v2.path(), v3.path()] {
        // A then B: B is the last writer; the delete stands.
        let mut session = session_for(target);
        run(&mut session, &format!("APPLY PATCH \"{patch_a}\";"));
        run(&mut session, &format!("APPLY PATCH \"{patch_b}\";"));
        let ab = state(&mut session);

        // Remove A: replaying B alone must not change what B decided.
        run(&mut session, &format!("REMOVE PATCH \"{patch_a}\";"));
        let b_only = state(&mut session);

        // Fresh session, B then A: now A is the last writer.
        let mut session = session_for(target);
        run(&mut session, &format!("APPLY PATCH \"{patch_b}\";"));
        run(&mut session, &format!("APPLY PATCH \"{patch_a}\";"));
        let ba = state(&mut session);

        outcomes.push((ab, b_only, ba));
    }

    assert_eq!(outcomes[0], outcomes[1], "stacking semantics diverge");
    let (ab, b_only, ba) = outcomes[0].clone();
    assert_eq!(ab.0.as_deref(), Some("[8]"), "last writer wins: {ab:?}");
    assert!(!ab.1, "the delete stands under A+B");
    assert_eq!(b_only.0.as_deref(), Some("[8]"), "B alone keeps B's write");
    assert!(!b_only.1);
    assert_eq!(
        ba.0.as_deref(),
        Some("[7]"),
        "reversed order flips the winner"
    );
    assert!(!ba.1, "the delete is order-independent here");
}

// ── The patch-algebra gates ──────────────────────────────────────────────────
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
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "parity/dense".into(),
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

/// insert → delete = absent; removing the delete patch resurrects the
/// inserted fact; removing the insert too returns to base. The delete
/// is a *fact* ("this entity has no KNN entries"), so removal replays
/// visibility rather than un-deleting storage.
#[test]
fn knn_delete_patch_removal_resurrects_the_inserted_fact() {
    let patch_dir = tempfile::tempdir().unwrap();
    let insert_patch = lql_path(&patch_dir.path().join("insert.vlp"));
    let delete_patch_file = patch_dir.path().join("delete.vlp");
    let delete_patch = lql_path(&delete_patch_file);

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{insert_patch}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
        );
        run(&mut session, "SAVE PATCH;");
    }
    write_patch(
        &delete_patch_file,
        vec![larql_vindex::PatchOp::DeleteKnn {
            entity: "[2]".into(),
        }],
    );

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut outcomes = Vec::new();
    for target in [v2.path(), v3.path()] {
        let mut session = session_for(target);
        let base = knn_fact_visible(&mut session);
        run(&mut session, &format!("APPLY PATCH \"{insert_patch}\";"));
        let inserted = knn_fact_visible(&mut session);
        run(&mut session, &format!("APPLY PATCH \"{delete_patch}\";"));
        let deleted = knn_fact_visible(&mut session);
        run(&mut session, &format!("REMOVE PATCH \"{delete_patch}\";"));
        let resurrected = knn_fact_visible(&mut session);
        run(&mut session, &format!("REMOVE PATCH \"{insert_patch}\";"));
        let emptied = knn_fact_visible(&mut session);
        outcomes.push((base, inserted, deleted, resurrected, emptied));
    }

    assert_eq!(outcomes[0], outcomes[1], "KNN patch algebra diverges");
    assert_eq!(
        outcomes[0],
        (false, true, false, true, false),
        "fold(base, active patches) must drive visibility: {:?}",
        outcomes[0]
    );
}

/// The logical-fingerprint gate: after applying and then removing
/// every patch, the ENTIRE feature space equals the base exactly —
/// the affected slots are restored AND no unaffected slot was dirtied
/// along the way. This catches mutation that has the desired semantic
/// effect but leaks state elsewhere.
#[test]
fn removing_every_patch_restores_the_exact_base_space() {
    let patch_dir = tempfile::tempdir().unwrap();
    let update_patch = lql_path(&patch_dir.path().join("u.vlp"));
    let delete_patch = lql_path(&patch_dir.path().join("d.vlp"));

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{update_patch}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{delete_patch}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut restored_spaces = Vec::new();
    for target in [v2.path(), v3.path()] {
        let mut session = session_for(target);
        let base = feature_space(&mut session, DENSE_LAYERS, 300);

        run(&mut session, &format!("APPLY PATCH \"{update_patch}\";"));
        run(&mut session, &format!("APPLY PATCH \"{delete_patch}\";"));
        let mutated = feature_space(&mut session, DENSE_LAYERS, 300);
        assert_ne!(base, mutated, "affirmative control: the patches must bite");

        run(&mut session, &format!("REMOVE PATCH \"{update_patch}\";"));
        run(&mut session, &format!("REMOVE PATCH \"{delete_patch}\";"));
        let restored = feature_space(&mut session, DENSE_LAYERS, 300);
        assert_eq!(
            base, restored,
            "removal must restore the exact base space — affected slots \
             back, unaffected slots never dirtied"
        );
        restored_spaces.push(restored);
    }
    assert_eq!(
        restored_spaces[0], restored_spaces[1],
        "restored spaces diverge across formats"
    );
}

/// Patches touching DISJOINT objects commute: every application order
/// of {KNN insert, slot update, slot delete} yields the same visible
/// state, on both backends. (Same-object precedence — last applied
/// wins — is gated in `patch_stacking_replays_in_order_on_both_backends`;
/// together they pin fold-order determinism.)
#[test]
fn disjoint_patches_commute_under_every_application_order() {
    let patch_dir = tempfile::tempdir().unwrap();
    let knn = lql_path(&patch_dir.path().join("k.vlp"));
    let upd = lql_path(&patch_dir.path().join("u.vlp"));
    let del = lql_path(&patch_dir.path().join("d.vlp"));

    {
        let author = v2_vindex();
        let mut session = session_for(author.path());
        run(&mut session, &format!("BEGIN PATCH \"{knn}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[2]", "b", "[5]");"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{upd}\";"));
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[7]" WHERE layer = 1 AND feature = 1;"#,
        );
        run(&mut session, "SAVE PATCH;");
        run(&mut session, &format!("BEGIN PATCH \"{del}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 1;",
        );
        run(&mut session, "SAVE PATCH;");
    }

    let orders: [[&String; 3]; 6] = [
        [&knn, &upd, &del],
        [&knn, &del, &upd],
        [&upd, &knn, &del],
        [&upd, &del, &knn],
        [&del, &knn, &upd],
        [&del, &upd, &knn],
    ];

    let v2 = v2_vindex();
    let v3 = v3_container();
    for target in [v2.path(), v3.path()] {
        let mut states = Vec::new();
        for order in &orders {
            let mut session = session_for(target);
            for patch in order {
                run(&mut session, &format!("APPLY PATCH \"{patch}\";"));
            }
            let space = feature_space(&mut session, DENSE_LAYERS, 300);
            let fact = knn_fact_visible(&mut session);
            states.push((fact, space));
        }
        assert!(states[0].0, "the KNN fact must be visible in every order");
        for (i, state) in states.iter().enumerate() {
            assert_eq!(
                &states[0], state,
                "order {i} diverged — disjoint patches must commute"
            );
        }
    }
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
        "parity/dense",
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
        "parity-dense",
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

/// The compose half of the mutation parity claim, staged so the first
/// divergence names its stage:
///
/// 1. **capture** — the two engines' residual statistic (the normed
///    FFN input, V2's walk-trace tap) agrees to cos ≥ 1 − 1e-5;
/// 2. **identity** — slots, layers, entities, targets, ids: EXACT;
/// 3. **magnitudes** — the reference norms are computed from the same
///    bytes with the same statistic, so vector norms agree within
///    0.1%. For the down column this doubles as an exact
///    balance-decision proxy: one diverged ×1.6/×0.7 step would shift
///    the norm ≥ 40% (observed agreement: 7+ digits — the two arms
///    took identical amplify/shrink/cross-fact sequences);
/// 4. **directions** — cos ≥ 1 − 1e-5 for all three vectors. The
///    refine path earns this tightness because the suppress-basis
///    RANK is stable: this gate's first run caught bitwise-duplicate
///    decoy residuals (degenerate fixture vocab) whose cross-arm
///    noise straddled Gram-Schmidt's 1e-6 near-dependency threshold,
///    flipping basis rank per arm and swinging refined directions to
///    cos ~0.98. With distinct decoy prompts (the real-model regime)
///    the only substrate delta is the stage-1-gated 1e-7 capture
///    noise through shared math.
#[test]
fn v2_and_v3_compose_installs_agree() {
    // ── Stage 1: capture parity, both install layers ──
    {
        let v2 = v2_vindex_worded();
        let v3 = v3_container_worded();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(v2.path(), &mut cb).unwrap();
        let tokenizer = larql_vindex::load_vindex_tokenizer(v2.path()).unwrap();
        let index = larql_vindex::VectorIndex::load_vindex(v2.path(), &mut cb).unwrap();
        let ids: Vec<u32> = tokenizer
            .encode("The b of a is", true)
            .unwrap()
            .get_ids()
            .to_vec();
        let walk = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, &index);
        let _ = larql_inference::predict_with_ffn(&weights, &tokenizer, &ids, 1, &walk);
        let v2_res = walk.take_residuals();

        use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
        let runtime = larql_inference::vindex3::Vindex3Runtime::open(
            v3.path(),
            "target",
            ProductionBackend::new(),
        )
        .unwrap();
        let mut v3_res: Vec<(usize, Vec<f32>)> = Vec::new();
        let continuation = runtime
            .select_continuation(
                &larql_kv::shipped_continuations(),
                &larql_vindex::format::vindex3::opplan::exec::kv::RowKvState::identity(),
                &larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig::empty(),
            )
            .unwrap();
        runtime
            .execute_streaming(&ids, &continuation, &mut |ev| {
                if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = ev {
                    v3_res.push((index, trace.ffn_input.last().unwrap().clone()));
                }
                Ok(())
            })
            .unwrap();
        for (layer, r2) in &v2_res {
            let r3 = &v3_res.iter().find(|(l, _)| l == layer).unwrap().1;
            let cos = cosine(r2, r3);
            assert!(
                cos >= 1.0 - 1e-5,
                "stage 1: capture diverges at layer {layer}: cos {cos}"
            );
        }
    }

    let patch_dir = tempfile::tempdir().unwrap();
    let script = |session: &mut Session, patch: &str| {
        run(session, &format!("BEGIN PATCH \"{patch}\";"));
        run(
            session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
        );
        run(
            session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
        );
        run(session, "SAVE PATCH;");
    };

    let v2 = v2_vindex_worded();
    let v2_patch = lql_path(&patch_dir.path().join("v2.vlp"));
    let mut v2_session = session_for(v2.path());
    script(&mut v2_session, &v2_patch);

    let v3 = v3_container_worded();
    let v3_patch = lql_path(&patch_dir.path().join("v3.vlp"));
    let mut v3_session = session_for(v3.path());
    script(&mut v3_session, &v3_patch);

    let load = |p: &str| larql_vindex::VindexPatch::load(std::path::Path::new(p)).unwrap();
    let (p2, p3) = (load(&v2_patch), load(&v3_patch));
    assert_eq!(p2.operations.len(), p3.operations.len(), "op counts");

    for (a, b) in p2.operations.iter().zip(&p3.operations) {
        let larql_vindex::PatchOp::Insert {
            layer: l2,
            feature: f2,
            entity: e2,
            target: t2,
            gate_vector_b64: g2,
            up_vector_b64: u2,
            down_vector_b64: d2,
            down_meta: m2,
            ..
        } = a
        else {
            panic!("V2 arm emitted a non-Insert op: {a:?}")
        };
        let larql_vindex::PatchOp::Insert {
            layer: l3,
            feature: f3,
            entity: e3,
            target: t3,
            gate_vector_b64: g3,
            up_vector_b64: u3,
            down_vector_b64: d3,
            down_meta: m3,
            ..
        } = b
        else {
            panic!("V3 arm emitted a non-Insert op: {b:?}")
        };
        assert_eq!((l2, f2, e2, t2), (l3, f3, e3, t3), "slot identity");
        assert_eq!(
            m2.as_ref().map(|m| m.top_token_id),
            m3.as_ref().map(|m| m.top_token_id),
            "target id"
        );
        for (name, min_cos, x, y) in [
            ("gate", 1.0f32 - 1e-5, g2, g3),
            ("up", 1.0 - 1e-5, u2, u3),
            ("down", 1.0 - 1e-5, d2, d3),
        ] {
            let (x, y) = (x.as_ref().unwrap(), y.as_ref().unwrap());
            let vx = larql_vindex::patch::core::decode_gate_vector(x).unwrap();
            let vy = larql_vindex::patch::core::decode_gate_vector(y).unwrap();
            let cos = cosine(&vx, &vy);
            let (nx, ny) = (
                vx.iter().map(|v| v * v).sum::<f32>().sqrt(),
                vy.iter().map(|v| v * v).sum::<f32>().sqrt(),
            );
            assert!(
                cos >= min_cos,
                "stage 4: {name} direction diverges for {e2}: cos {cos} (norms {nx} vs {ny})"
            );
            assert!(
                (nx - ny).abs() <= 1e-3 * nx.max(ny).max(1e-12),
                "stage 3: {name} magnitude diverges for {e2}: {nx} vs {ny}"
            );
        }
    }
}

/// DIFF across generations refuses with direction, never a confused
/// half-comparison: realise both models in one generation first.
#[test]
fn diff_across_generations_refuses_with_direction() {
    let v2 = v2_vindex();
    let v3 = v3_container();
    let mut session = Session::new();
    let stmt = format!(
        "DIFF \"{}\" \"{}\";",
        lql_path(v2.path()),
        lql_path(v3.path())
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("mixed-generation diff must refuse");
    assert!(err.to_string().contains("across generations"), "{err}");
}

/// `PHYSICAL` is a VINDEX3 report — V2 sides refuse it with direction.
#[test]
fn physical_diff_refuses_on_v2_sides() {
    let v2 = v2_vindex();
    let mut session = Session::new();
    let stmt = format!(
        "DIFF \"{}\" \"{}\" PHYSICAL;",
        lql_path(v2.path()),
        lql_path(v2.path())
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("V2 sides must refuse PHYSICAL");
    assert!(err.to_string().contains("VINDEX3 report"), "{err}");
}

/// `COMPACT INTO VINDEX` is the V3 physical statement — a V2 binding
/// is directed to its own tiered compaction.
#[test]
fn compact_into_vindex_refuses_on_v2_with_direction() {
    let v2 = v2_vindex();
    let mut session = session_for(v2.path());
    let err = session
        .execute(&parse(r#"COMPACT INTO VINDEX "out.v3";"#).unwrap())
        .expect_err("V2 must refuse the V3 physical compact");
    assert!(err.to_string().contains("COMPACT MINOR"), "{err}");
}
