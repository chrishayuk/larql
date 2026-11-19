//! V3-LQL-3B gates: mutation on a VINDEX3 binding, closed-loop.
//!
//! The oracle is stronger than "INSERT returns OK": every claim is a
//! **round trip through the statement surface**.
//!
//! ```text
//! before:  DESCRIBE → edge absent;  INFER → no override
//! INSERT MODE KNN (the default)
//! after:   DESCRIBE → edge present; INFER → stored target overrides top-1
//! SAVE PATCH
//! reopen pristine container → absent again
//! APPLY PATCH → present again
//! REMOVE PATCH → absent again
//! ```
//!
//! The KNN key is captured from the V3 runtime's own execution (plan
//! taps), so same-prompt retrieval is exact by construction — the
//! same property the V2 arm gets from its forward pass.

use std::path::Path;

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// The canonical prompt `INSERT ("a", "b", …)` captures its key from —
/// INFER on this exact prompt must retrieve the stored target.
const CANONICAL_PROMPT: &str = "The b of a is";

/// Windows temp paths contain backslashes, which the LQL lexer's escape
/// pass would consume; doubling them leaves the path untouched on every
/// platform.
fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

/// The synthetic `[N]` ↔ id N tokenizer, extended with spaced forms
/// (`" [N]"` ↔ id N): the V2 target contract encodes `" {target}"`,
/// and without a pre-tokenizer the spaced surface must be in-vocab or
/// every target degrades to `[UNK]` (a ~zero embedding — which turned
/// a compose install's payload into a no-op the first time this ran).
fn spaced_tokenizer_json(vocab: usize) -> String {
    let mut entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    entries.extend((0..vocab).map(|i| format!("\" [{i}]\":{i}")));
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "mutation-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        spaced_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn bound_session(container: &Path) -> Session {
    let mut session = Session::new();
    let use_stmt = format!("USE \"{}\";", lql_path(container));
    run(&mut session, &use_stmt);
    session
}

fn describe_entity(session: &mut Session) -> String {
    run(session, r#"DESCRIBE "a";"#).join("\n")
}

fn infer_canonical(session: &mut Session) -> String {
    run(session, &format!("INFER \"{CANONICAL_PROMPT}\" TOP 3;")).join("\n")
}

/// The core closed loop: absent → INSERT → present, observed through
/// DESCRIBE **and** through INFER's post-logits override.
#[test]
fn insert_knn_round_trips_through_describe_and_infer() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    // Pre-screen: the edge must be genuinely absent before the insert,
    // or "present after" proves nothing.
    let before = describe_entity(&mut session);
    assert!(!before.contains("→ [5]"), "pristine container: {before}");
    let infer_before = infer_canonical(&mut session);
    assert!(
        !infer_before.contains("knn_override"),
        "pristine container: {infer_before}"
    );

    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    )
    .join("\n");
    assert!(out.contains("Inserted: a —[b]→ [5]"), "{out}");
    assert!(out.contains("KNN store: 1 entries total"), "{out}");
    assert!(out.contains("VINDEX3 plan taps"), "{out}");

    // The overlay is immediately visible to browse…
    let after = describe_entity(&mut session);
    assert!(after.contains("→ [5]"), "{after}");

    // …and to inference: same-prompt retrieval fires the shared
    // post-logits gate and the stored target takes row 1.
    let infer_after = infer_canonical(&mut session);
    assert!(infer_after.contains("knn_override"), "{infer_after}");
    let row1 = infer_after
        .lines()
        .find(|l| l.trim_start().starts_with("1."))
        .unwrap_or_else(|| panic!("no row 1 in {infer_after}"));
    assert!(row1.contains("[5]"), "stored target must lead: {row1}");
    assert!(
        infer_after.contains("post-logits retrieval sidecar"),
        "{infer_after}"
    );
}

/// The full patch lifecycle: the mutation persists as a portable patch,
/// a pristine reopen loses it, APPLY restores it, REMOVE drops it.
#[test]
fn patch_lifecycle_round_trips_on_a_pristine_reopen() {
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = patch_dir.path().join("facts.vlp");
    let patch_stmt_path = lql_path(&patch_file);

    // Session 1: record the mutation into a named patch.
    {
        let mut session = bound_session(container.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_stmt_path}\";"));
        run(
            &mut session,
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
        );
        let saved = run(&mut session, "SAVE PATCH;").join("\n");
        assert!(saved.contains("Saved:"), "{saved}");
        assert!(saved.contains("1 inserts"), "{saved}");
    }
    assert!(patch_file.exists(), "SAVE PATCH must write the file");

    // Session 2: the pristine container knows nothing of the edit —
    // the base was never modified.
    let mut session = bound_session(container.path());
    let pristine = describe_entity(&mut session);
    assert!(!pristine.contains("→ [5]"), "{pristine}");
    assert!(!infer_canonical(&mut session).contains("knn_override"));

    // APPLY restores the logical fact from the portable patch.
    let applied = run(&mut session, &format!("APPLY PATCH \"{patch_stmt_path}\";")).join("\n");
    assert!(applied.contains("Applied:"), "{applied}");
    assert!(describe_entity(&mut session).contains("→ [5]"));
    assert!(infer_canonical(&mut session).contains("knn_override"));

    let listed = run(&mut session, "SHOW PATCHES;").join("\n");
    assert!(listed.contains("1 ops"), "{listed}");

    // REMOVE drops it again — the overlay rebuilds from the remaining
    // (empty) patch list.
    let removed = run(
        &mut session,
        &format!("REMOVE PATCH \"{patch_stmt_path}\";"),
    )
    .join("\n");
    assert!(removed.contains("Removed"), "{removed}");
    let after_remove = describe_entity(&mut session);
    assert!(!after_remove.contains("→ [5]"), "{after_remove}");
    assert!(!infer_canonical(&mut session).contains("knn_override"));
}

/// MERGE resolves the V3 binding as its target; a source directory
/// that is not a vindex fails at source loading with a helpful error —
/// never the misleading "no backend loaded".
#[test]
fn merge_with_an_invalid_source_reports_the_load_failure() {
    let container = v3_container();
    let source = tempfile::tempdir().unwrap();
    let mut session = bound_session(container.path());
    let stmt = format!("MERGE \"{}\";", lql_path(source.path()));
    let parsed = parse(&stmt).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("an empty dir is not a vindex");
    let msg = err.to_string();
    assert!(msg.contains("failed to load source"), "{msg}");
    assert!(!msg.contains("No backend"), "{msg}");
}

/// A tokenizerless container cannot capture the canonical prompt —
/// the refusal names the missing capability, and no entry lands.
#[test]
fn insert_refuses_on_a_tokenizerless_container() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "tokless-fixture",
    );
    let mut session = bound_session(container.path());
    let parsed =
        parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#).unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("INSERT needs the tokenizer capability");
    assert!(err.to_string().contains("tokenizer"), "{err}");
}

/// `AT LAYER n` pins the install layer instead of the default
/// penultimate layer — and out-of-range hints clamp, as on V2.
#[test]
fn insert_at_layer_pins_the_install_layer() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1;"#,
    )
    .join("\n");
    assert!(out.contains("at L1"), "{out}");

    let clamped = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("x", "y", "[6]") AT LAYER 99;"#,
    )
    .join("\n");
    assert!(clamped.contains("at L1"), "out-of-range clamps: {clamped}");
}

/// Read the feature ids SELECT reports for one layer.
fn feature_ids_at_layer(session: &mut Session, layer: usize) -> Vec<usize> {
    run(
        session,
        &format!("SELECT * FROM FEATURES WHERE layer = {layer} LIMIT 300;"),
    )
    .iter()
    .filter_map(|line| {
        let mut parts = line.split_whitespace();
        let (_l, f) = (parts.next()?, parts.next()?);
        f.strip_prefix('F')?.parse().ok()
    })
    .collect()
}

/// Feature-slot mutation closed loop (V3-LQL-3B rung 2): UPDATE
/// rewrites the annotation SELECT reports; DELETE tombstones the slot
/// out of SELECT; and — V2's statement-surface contract — an UPDATE
/// on a tombstoned slot finds nothing (its meta reads as absent), so
/// resurrection happens through patch replay, not through UPDATE.
#[test]
fn update_rewrites_and_delete_tombstones_through_select() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    let before = feature_ids_at_layer(&mut session, 0);
    assert!(before.contains(&0), "fixture must annotate feature 0");

    // UPDATE a live slot: the new annotation is what SELECT reports.
    let out = run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 0 AND feature = 0;"#,
    )
    .join("\n");
    assert!(out.contains("Updated 1 features"), "{out}");
    let rows = run(
        &mut session,
        "SELECT * FROM FEATURES WHERE layer = 0 LIMIT 300;",
    )
    .join("\n");
    let row0 = rows
        .lines()
        .find(|l| l.split_whitespace().nth(1) == Some("F0"))
        .unwrap_or_else(|| panic!("no F0 row in {rows}"));
    assert!(row0.contains("[9]"), "{row0}");

    // DELETE tombstones it out of the feature space.
    let out = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .join("\n");
    assert!(out.contains("Deleted 1 features"), "{out}");
    let after_delete = feature_ids_at_layer(&mut session, 0);
    assert!(!after_delete.contains(&0), "tombstoned slot must vanish");
    assert_eq!(after_delete.len(), before.len() - 1, "only that slot");

    // V2 parity: UPDATE cannot see a tombstoned slot's meta, so it
    // matches nothing — the same answer a V2 session gives.
    let out = run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 0 AND feature = 0;"#,
    )
    .join("\n");
    assert!(out.contains("no matching features"), "{out}");
}

/// UPDATE reads the current (overlay-merged) meta, so a second UPDATE
/// composes on the first — and WALK observes the tombstone filter.
#[test]
fn walk_excludes_tombstoned_features() {
    let container = v3_container();
    let mut session = bound_session(container.path());

    // Find the top walk hit, delete exactly that slot, walk again.
    let walk_before = run(&mut session, r#"WALK "[3]" TOP 3;"#).join("\n");
    let hit = walk_before
        .lines()
        .find_map(|l| {
            let t = l.trim_start().strip_prefix("L")?;
            let (layer, rest) = t.split_once(':')?;
            let feat = rest
                .trim_start()
                .strip_prefix('F')?
                .split_whitespace()
                .next()?;
            Some((
                layer.trim().parse::<usize>().ok()?,
                feat.parse::<usize>().ok()?,
            ))
        })
        .expect("walk must return a hit");

    run(
        &mut session,
        &format!(
            "DELETE FROM EDGES WHERE layer = {} AND feature = {};",
            hit.0, hit.1
        ),
    );
    let walk_after = run(&mut session, r#"WALK "[3]" TOP 3;"#).join("\n");
    let needle = format!("F{}", hit.1);
    let still_there = walk_after
        .lines()
        .any(|l| l.trim_start().starts_with(&format!("L {}:", hit.0)) && l.contains(&needle));
    assert!(
        !still_there,
        "tombstoned hit must leave the walk:\n{walk_after}"
    );
}

/// The feature-slot patch lifecycle: DELETE + UPDATE persist as a
/// portable patch, a pristine reopen loses them, APPLY restores them,
/// REMOVE drops them.
#[test]
fn feature_patch_lifecycle_round_trips_on_a_pristine_reopen() {
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = patch_dir.path().join("edits.vlp");
    let patch_stmt_path = lql_path(&patch_file);

    {
        let mut session = bound_session(container.path());
        run(&mut session, &format!("BEGIN PATCH \"{patch_stmt_path}\";"));
        run(
            &mut session,
            "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
        );
        run(
            &mut session,
            r#"UPDATE EDGES SET target = "[9]" WHERE layer = 1 AND feature = 1;"#,
        );
        let saved = run(&mut session, "SAVE PATCH;").join("\n");
        assert!(saved.contains("1 updates, 1 deletes"), "{saved}");
    }

    let mut session = bound_session(container.path());
    assert!(
        feature_ids_at_layer(&mut session, 0).contains(&0),
        "pristine reopen must not carry the delete"
    );

    run(&mut session, &format!("APPLY PATCH \"{patch_stmt_path}\";"));
    assert!(!feature_ids_at_layer(&mut session, 0).contains(&0));
    let rows = run(
        &mut session,
        "SELECT * FROM FEATURES WHERE layer = 1 LIMIT 300;",
    )
    .join("\n");
    let row = rows
        .lines()
        .find(|l| l.split_whitespace().nth(1) == Some("F1"))
        .unwrap_or_else(|| panic!("no F1 row in {rows}"));
    assert!(row.contains("[9]"), "{row}");

    run(
        &mut session,
        &format!("REMOVE PATCH \"{patch_stmt_path}\";"),
    );
    assert!(
        feature_ids_at_layer(&mut session, 0).contains(&0),
        "REMOVE must restore the pristine feature space"
    );
}

/// INFER's output normalised for exact comparison: timing stripped,
/// and prediction rows reduced to `rank (prob) [id N]`. The decoded
/// surface is dropped because the spaced-vocab fixture maps two
/// surfaces onto one id and WordLevel decode picks one untracked —
/// tokenizer cosmetics, not program output (the 3A lesson about
/// ambiguous fixture vocabularies, resurfacing on the decode side).
fn infer_lines(session: &mut Session, prompt: &str) -> Vec<String> {
    run(session, &format!("INFER \"{prompt}\" TOP 3;"))
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .map(|l| {
            let trimmed = l.trim_start();
            match (trimmed.split_once(". "), l.rfind('(')) {
                (Some((rank, _)), Some(paren)) if rank.chars().all(|c| c.is_ascii_digit()) => {
                    format!("{rank}. {}", &l[paren..])
                }
                _ => l,
            }
        })
        .collect()
}

/// The compose closed loop (V3-LQL-3B compose) — the operand-source
/// seam's oracle:
///
/// ```text
/// before:  INFER = baseline
/// compose INSERT
/// after:   browse sees the slot; execution reads the overridden
///          operands; INFER promotes the stored target; TRACE agrees
/// SAVE PATCH → pristine reopen = baseline, bit for bit
/// APPLY PATCH → the composed outputs return, bit for bit
/// REMOVE PATCH → baseline again, bit for bit
/// ```
#[test]
fn compose_insert_alters_execution_and_reverts_bit_for_bit() {
    // Entity and relation are REAL fixture tokens: the capture prompt
    // "The [7] of [3] is" must carry actual embedding signal into the
    // captured residual. With [UNK]-only words (`"a"`, `"b"`) the
    // residual entering the RMS norm is rounding noise, the norm
    // amplifies it to an O(1) but platform-random direction, and the
    // walk-ranking assertion below becomes a coin toss (it lost on
    // Windows CI while passing on unix).
    const COMPOSE_PROMPT: &str = "The [7] of [3] is";
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = lql_path(patch_dir.path().join("compose.vlp"));

    let mut session = bound_session(container.path());
    let baseline = infer_lines(&mut session, COMPOSE_PROMPT);
    assert!(
        !baseline.join("\n").contains("\"[5]\""),
        "pre-screen: the target must not already lead: {baseline:?}"
    );

    // ── Install, with a payload strong enough to flip top-1 ──
    // ALPHA 5000: the LCG fixture's head is random, so the payload's
    // projection onto any logit is tiny and the balance loop never
    // reaches PROB_FLOOR to amplify it. At ALPHA 5.0 the change to
    // INFER's 2-decimal display was sub-quantum — whether a digit
    // flipped was platform rounding luck (it did on unix, not on
    // Windows CI). 5000 makes the observed difference a top-1
    // reordering with ~8x the display quantum in probability margin.
    run(&mut session, &format!("BEGIN PATCH \"{patch_file}\";"));
    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") ALPHA 5000.0 MODE COMPOSE;"#,
    )
    .join("\n");
    assert!(out.contains("compose overlay"), "{out}");
    let slot = out
        .lines()
        .find_map(|l| l.split(" at L").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .unwrap_or_else(|| panic!("no slot in {out}"))
        .to_string();
    let (layer, feature) = slot
        .split_once("/F")
        .map(|(l, f)| (l.parse::<usize>().unwrap(), f.parse::<usize>().unwrap()))
        .unwrap_or_else(|| panic!("unparsable slot {slot}"));

    // Browse sees the slot: the annotation is in the feature space…
    let rows = run(
        &mut session,
        &format!("SELECT * FROM FEATURES WHERE layer = {layer} LIMIT 300;"),
    )
    .join("\n");
    let row = rows
        .lines()
        .find(|l| l.split_whitespace().nth(1) == Some(&format!("F{feature}")))
        .unwrap_or_else(|| panic!("no F{feature} row in {rows}"));
    assert!(row.contains("[5]"), "{row}");
    // …and the overridden gate row is merged into the scan: the
    // entity token's walk ranks the ×30 gate above the layer's trained
    // rows. This needs the conditioned capture above — the composed
    // gate is `unit(residual)·g_ref·30`, and only a residual carrying
    // the entity's embedding gives it a stable response to "[3]".
    let walk = run(&mut session, r#"WALK "[3]" TOP 5;"#).join("\n");
    assert!(
        walk.contains(&format!("F{feature}")),
        "the composed slot must surface in the walk:\n{walk}"
    );

    // Execution observes the edit: the effective program differs from
    // the stored one. (Target *promotion* — the payload lifting [5] to
    // top-1 — needs an output head aligned with the embedding table,
    // which V2 validated on Gemma; this LCG fixture's head is random,
    // so the honest claim here is observation + reversion, not
    // steering quality.)
    let composed = infer_lines(&mut session, COMPOSE_PROMPT);
    assert_ne!(baseline, composed, "the install must change execution");

    // TRACE runs the same effective program — no fork between the
    // observed executor and INFER.
    let row1_id: u32 = composed
        .iter()
        .find(|l| l.trim_start().starts_with("1."))
        .and_then(|l| l.split("[id ").nth(1))
        .and_then(|s| s.trim_end_matches(']').trim().parse().ok())
        .unwrap_or_else(|| panic!("no row-1 id in {composed:?}"));
    let trace = run(&mut session, &format!("TRACE \"{COMPOSE_PROMPT}\";")).join("\n");
    let traced_next = trace
        .lines()
        .find(|l| l.starts_with("next token"))
        .unwrap_or_else(|| panic!("no next-token line in {trace}"));
    assert!(
        traced_next.contains(&format!("next token {row1_id} ")),
        "TRACE and INFER must run the same effective program: {traced_next} vs id {row1_id}"
    );

    let saved = run(&mut session, "SAVE PATCH;").join("\n");
    assert!(saved.contains("1 inserts"), "{saved}");

    // ── Pristine reopen: the container is untouched ──
    let mut session = bound_session(container.path());
    assert_eq!(
        infer_lines(&mut session, COMPOSE_PROMPT),
        baseline,
        "a fresh open must be bit-for-bit baseline"
    );

    // APPLY: the composed behaviour returns, bit for bit.
    run(&mut session, &format!("APPLY PATCH \"{patch_file}\";"));
    assert_eq!(
        infer_lines(&mut session, COMPOSE_PROMPT),
        composed,
        "replaying the patch must reproduce the composed outputs"
    );

    // REMOVE: baseline again, bit for bit.
    run(&mut session, &format!("REMOVE PATCH \"{patch_file}\";"));
    assert_eq!(infer_lines(&mut session, COMPOSE_PROMPT), baseline);
}

/// Compose INSERT honours `AT LAYER` and refuses without a tokenizer,
/// like the KNN arm.
#[test]
fn compose_insert_pins_the_layer_and_needs_the_tokenizer() {
    let container = v3_container();
    let mut session = bound_session(container.path());
    let out = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    )
    .join("\n");
    assert!(out.contains("at L1/"), "{out}");

    // Exhaust the layer's slots: every install claims one, and the
    // 21st (the miniature has 20 features) reports the exhaustion.
    for i in 1..20 {
        run(
            &mut session,
            &format!(
                "INSERT INTO EDGES (entity, relation, target) VALUES (\"e{i}\", \"b\", \"[5]\") AT LAYER 1 MODE COMPOSE;"
            ),
        );
    }
    let parsed = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("late", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    )
    .unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("a full layer must refuse the install");
    assert!(err.to_string().contains("no free feature slot"), "{err}");

    let checkpoint = tempfile::tempdir().unwrap();
    let tokless = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        tokless.path(),
        "tokless-compose",
    );
    let mut session = bound_session(tokless.path());
    let parsed = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") MODE COMPOSE;"#,
    )
    .unwrap();
    let err = session
        .execute(&parsed)
        .expect_err("compose needs the tokenizer capability");
    assert!(err.to_string().contains("tokenizer"), "{err}");
}

/// REBALANCE on V3: the fixed-point loop runs over the composed
/// program. A wide band converges instantly (every fact in band). After
/// the container is bound again, the session holds no installs (Gate B,
/// B1): REBALANCE has nothing to rebalance and says so, as in a fresh
/// session. It used to keep probing the previous binding's installs.
#[test]
fn rebalance_converges_and_a_rebind_leaves_nothing_to_rebalance() {
    let container = v3_container();
    let patch_dir = tempfile::tempdir().unwrap();
    let patch_file = lql_path(patch_dir.path().join("rb.vlp"));
    let mut session = bound_session(container.path());
    run(&mut session, &format!("BEGIN PATCH \"{patch_file}\";"));
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );

    // Wide band: every fact is in band on the first probe — the loop
    // converges in one iteration.
    let out = run(&mut session, "REBALANCE MAX 4 FLOOR 0.0 CEILING 1.0;").join("\n");
    assert!(out.contains("1 compose installs"), "{out}");
    assert!(out.contains("all converged in band"), "{out}");

    run(&mut session, "SAVE PATCH;");
    run(
        &mut session,
        &format!("USE \"{}\";", lql_path(container.path())),
    );
    let out = run(&mut session, "REBALANCE MAX 2 FLOOR 0.0 CEILING 1.0;").join("\n");
    let mut fresh = bound_session(container.path());
    let want = run(&mut fresh, "REBALANCE MAX 2 FLOOR 0.0 CEILING 1.0;").join("\n");
    assert!(out.contains("no compose-mode installs"), "{out}");
    assert_eq!(out, want);
}
