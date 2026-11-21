//! Gate B, B6 §3.1 (ruling D1 = P): COMPACT MINOR promotes each KNN fact to
//! a keyed compose install, recorded as `delete_knn_fact(K)` then the keyed
//! compose `Insert` for K, as one atomic statement.
//!
//! The fixture's KNN facts sit in an applied patch (`base.vlp`), so REMOVE
//! PATCH of the promotion is allowed (B2). A session applies the base,
//! opens a recording (`promo.vlp`), runs COMPACT MINOR and saves.
//!
//! The six required tests (§3.1) are, in order: identity, replay, a failure
//! between delete and install, APPLY, REMOVE PATCH, and a second attempt.
//! The refusals before any change follow.

use std::path::{Path, PathBuf};

use larql_lql::install_fault::InstallFaultPoint as P;
use larql_lql::{AssertedFact, FactRealization, FactStatus, Session};
use larql_vindex::KnnStore;

use crate::support::*;

/// The facts promoted: two KNN facts at two layers, inserted in the
/// reverse of the order COMPACT MINOR must promote them in.
const BASE: [&str; 2] = [
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[4]", "[7]", "[6]") AT LAYER 1 MODE KNN;"#,
    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "[7]", "[5]") AT LAYER 0 MODE KNN;"#,
];
const PROMOTED: usize = BASE.len();

struct Fixture {
    dir: tempfile::TempDir,
    source: PathBuf,
    base: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let source = v2_vindex(dir.path());
        let base = dir.path().join("base.vlp");
        let mut session = bound(&source);
        run(
            &mut session,
            &format!("BEGIN PATCH \"{}\";", lql_path(&base)),
        );
        for stmt in BASE {
            run(&mut session, stmt);
        }
        run(&mut session, "SAVE PATCH;");
        Self { dir, source, base }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// A session bound to the source with the base patch applied.
    fn based(&self) -> Session {
        let mut session = bound(&self.source);
        run(
            &mut session,
            &format!("APPLY PATCH \"{}\";", lql_path(&self.base)),
        );
        session
    }

    /// [`Self::based`] with a recording open into `name`.
    fn recording(&self, name: &str) -> Session {
        let mut session = self.based();
        run(
            &mut session,
            &format!("BEGIN PATCH \"{}\";", lql_path(self.path(name))),
        );
        session
    }

    /// A session that promoted and saved `name`.
    fn promoted(&self, name: &str) -> (Session, Vec<String>) {
        let mut session = self.recording(name);
        let out = run(&mut session, "COMPACT MINOR;");
        run(&mut session, "SAVE PATCH;");
        (session, out)
    }
}

fn facts(session: &Session) -> Vec<AssertedFact> {
    session.asserted_facts().unwrap()
}

/// The KNN store's entries, layer by layer and in order, as text. The key
/// rows are derived from them (and held in a hash map whose print order is
/// arbitrary), so the entries are what is compared.
fn knn(session: &mut Session) -> String {
    format!(
        "{:?}",
        session.patched_overlay_mut().unwrap().knn_store.entries()
    )
}

/// What a replay must reproduce bit for bit: the override vectors, the KNN
/// store, and each fact's identity, representation, status and slots.
fn logical(session: &mut Session) -> String {
    let overrides = state(session).split("\n--\n").nth(1).unwrap().to_string();
    let facts: Vec<_> = facts(session)
        .into_iter()
        .map(|f| {
            (
                f.id,
                f.entity,
                f.relation,
                f.target,
                f.realization,
                f.status,
                f.layers,
                f.slots,
            )
        })
        .collect();
    format!("{overrides}\n{}\n{facts:?}", knn(session))
}

/// The catalogue comparison (§3.1): the same ids with the same key and
/// target; each promoted fact changed only from KNN to compose; every other
/// fact unchanged; none ambiguous, inconsistent or degraded.
fn assert_catalogue_promoted(before: &[AssertedFact], after: &[AssertedFact]) {
    let ids = |fs: &[AssertedFact]| fs.iter().map(|f| f.id.clone()).collect::<Vec<_>>();
    assert_eq!(
        ids(before),
        ids(after),
        "a fact disappeared or gained an id"
    );
    for (b, a) in before.iter().zip(after) {
        assert_eq!(
            (&b.entity, &b.relation, &b.target),
            (&a.entity, &a.relation, &a.target),
            "{}: key or target changed",
            b.id
        );
        assert!(
            !matches!(
                a.status,
                FactStatus::Ambiguous | FactStatus::Inconsistent | FactStatus::Degraded
            ),
            "{}: {:?}",
            a.id,
            a.status
        );
        if b.realization == FactRealization::Knn {
            assert_eq!(a.realization, FactRealization::Compose, "{}", a.id);
            assert_eq!(a.status, FactStatus::Compose, "{}", a.id);
            assert_eq!(a.layers, b.layers, "{}: installed at another layer", a.id);
        } else {
            assert_eq!(a, b, "a fact COMPACT MINOR did not promote changed");
        }
    }
}

/// Test 1: identity.
#[test]
fn compact_minor_keeps_each_fact_identity() {
    let fx = Fixture::new();
    let mut session = fx.recording("promo.vlp");
    let before = facts(&session);
    assert_eq!(
        before
            .iter()
            .filter(|f| f.realization == FactRealization::Knn)
            .count(),
        PROMOTED
    );
    let out = run(&mut session, "COMPACT MINOR;");
    assert_catalogue_promoted(&before, &facts(&session));
    assert_eq!(knn(&mut session), "{}", "L0 is empty after promotion");
    let text = out.join("\n");
    for f in &before {
        assert!(text.contains(&f.id), "the output names {}: {text}", f.id);
    }
    assert!(!text.contains("failed"), "{text}");
    assert!(text.contains("lose their identity"), "C-9 note: {text}");
}

/// Test 1, ordering (R1): promotions run by layer, then fact key, whatever the
/// order the facts were inserted in; the recording holds the delete then
/// the keyed compose insert for each.
#[test]
fn compact_minor_records_delete_then_insert_in_a_defined_order() {
    let fx = Fixture::new();
    fx.promoted("promo.vlp");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fx.path("promo.vlp")).unwrap()).unwrap();
    let ops: Vec<(String, String)> = json["operations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|op| op["op"] != "update")
        .map(|op| {
            (
                op["op"].as_str().unwrap().to_string(),
                op["entity"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    let want = [
        ("delete_knn_fact", "[3]"),
        ("insert", "[3]"),
        ("delete_knn_fact", "[4]"),
        ("insert", "[4]"),
    ];
    let want: Vec<(String, String)> = want
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    assert_eq!(ops, want, "{json}");

    // A second, independent session records the same operations.
    fx.promoted("again.vlp");
    let again: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fx.path("again.vlp")).unwrap()).unwrap();
    assert_eq!(json["operations"], again["operations"]);
}

/// Tests 2 and 4: replay and APPLY. the saved promotion applies to a fresh
/// session over the same base without an I-2 refusal, and reaches the live
/// state bit for bit, and the same compiled output.
#[test]
fn compact_minor_replays_to_the_live_state() {
    let fx = Fixture::new();
    let (mut live, _) = fx.promoted("promo.vlp");
    let mut replay = fx.based();
    run(
        &mut replay,
        &format!("APPLY PATCH \"{}\";", lql_path(fx.path("promo.vlp"))),
    );
    assert_eq!(logical(&mut replay), logical(&mut live));
    assert_catalogue_promoted(&facts(&fx.based()), &facts(&replay));

    let compile = |s: &mut Session, name: &str| {
        let out = fx.path(name);
        run(
            s,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
        );
        out
    };
    assert_same_tree(
        &compile(&mut live, "live.out"),
        &compile(&mut replay, "replay.out"),
    );
}

/// Test 5: REMOVE PATCH of the promotion restores the KNN representation.
#[test]
fn removing_the_promotion_patch_restores_the_knn_facts() {
    let fx = Fixture::new();
    fx.promoted("promo.vlp");
    let mut base = fx.based();
    let mut session = fx.based();
    let promo = lql_path(fx.path("promo.vlp"));
    run(&mut session, &format!("APPLY PATCH \"{promo}\";"));
    run(&mut session, &format!("REMOVE PATCH \"{promo}\";"));
    assert_eq!(facts(&session), facts(&base));
    assert_eq!(knn(&mut session), knn(&mut base));
}

/// Test 6: a second COMPACT MINOR finds L0 empty and records nothing.
#[test]
fn a_second_compact_minor_never_duplicates_a_fact() {
    let fx = Fixture::new();
    let mut session = fx.recording("promo.vlp");
    run(&mut session, "COMPACT MINOR;");
    let before = state(&mut session);
    let out = run(&mut session, "COMPACT MINOR;");
    assert!(out.join("\n").contains("L0 is empty"), "{out:?}");
    assert_eq!(
        state(&mut session),
        before,
        "the second attempt changed the session"
    );
}

/// Test 3: a failure at any point, between a delete and its install included,
/// restores the whole previous state, with and without a recording open;
/// the retry then reaches what a session that never failed reaches.
#[test]
fn a_failed_promotion_restores_everything() {
    let fx = Fixture::new();
    let points = [
        P::CompactBeforePromotions,
        P::CompactAfterKnnRemoval { promotion: 1 },
        P::CompactAfterPromotion { promotion: 1 },
        P::CompactAfterKnnRemoval {
            promotion: PROMOTED,
        },
        P::CompactAfterPromotion {
            promotion: PROMOTED,
        },
        // Inside the first compose install (B5's own points).
        P::AfterRefine,
        P::BeforeRecord,
    ];
    for open in [true, false] {
        let start = |fx: &Fixture| {
            if open {
                fx.recording("promo.vlp")
            } else {
                fx.based()
            }
        };
        let mut reference = start(&fx);
        run(&mut reference, "COMPACT MINOR;");
        let reference = state(&mut reference);

        for point in points {
            let at = format!("{point} (recording open: {open})");
            let mut session = start(&fx);
            let before = (state(&mut session), knn(&mut session));
            session.set_install_fault(Some(point));
            let err = try_run(&mut session, "COMPACT MINOR;").expect_err(&at);
            session.set_install_fault(None);
            assert!(err.contains("rolled back"), "{at}: {err}");
            assert!(err.contains("injected fault"), "{at}: {err}");
            let after = (state(&mut session), knn(&mut session));
            assert!(after == before, "{at}: the session changed");

            run(&mut session, "COMPACT MINOR;");
            assert_eq!(state(&mut session), reference, "{at}: the retry differs");
        }
    }
}

/// Rewrites the source's own KNN store with `edit`, so a container carries
/// what no LQL statement would write today (a legacy ledger).
fn legacy_source(fx: &Fixture, edit: impl FnOnce(&mut KnnStore)) -> PathBuf {
    let out = fx.path("legacy.vindex");
    let mut session = fx.based();
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    );
    let file = out.join(larql_vindex::format::filenames::KNN_STORE_BIN);
    let mut store = KnnStore::load(&file).unwrap();
    edit(&mut store);
    store.save(&file).unwrap();
    // The edited store no longer matches the recorded checksum.
    let idx = out.join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&idx).unwrap()).unwrap();
    index["checksums"]
        .as_object_mut()
        .map(|c| c.remove(larql_vindex::format::filenames::KNN_STORE_BIN));
    std::fs::write(&idx, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    out
}

fn assert_refuses_without_change(source: &Path, reason: &str) {
    let mut session = bound(source);
    let before = (state(&mut session), knn(&mut session));
    let err = try_run(&mut session, "COMPACT MINOR;").expect_err("must refuse");
    assert!(err.contains("refused, nothing was changed"), "{err}");
    assert!(err.contains(reason), "{err}");
    let after = (state(&mut session), knn(&mut session));
    assert!(after == before, "a refusal changed the session");
}

/// A KNN fact that is not `active` is refused before any change: here a
/// target that needs more than one token.
#[test]
fn compact_minor_refuses_a_fact_that_is_not_active() {
    let fx = Fixture::new();
    let source = legacy_source(&fx, |store| {
        let (layer, entry) = store.ordered_entries()[0];
        let entry = entry.clone();
        store.add(
            layer,
            entry.key,
            entry.target_id,
            "[5] [6]".into(),
            "[9]".into(),
            "[7]".into(),
            entry.confidence,
        );
    });
    assert_refuses_without_change(&source, "multi_token");
}

/// A key live at two KNN layers (`ambiguous`, §3.1 test 6's "live twice"
/// case) is refused before any change, so it is never promoted twice.
#[test]
fn compact_minor_refuses_a_fact_live_at_two_layers() {
    let fx = Fixture::new();
    let source = legacy_source(&fx, |store| {
        let (layer, entry) = store.ordered_entries()[0];
        let entry = entry.clone();
        store.add(
            layer + 2,
            entry.key,
            entry.target_id,
            entry.target_token,
            entry.entity,
            entry.relation,
            entry.confidence,
        );
    });
    let fact = facts(&bound(&source))
        .into_iter()
        .find(|f| f.entity == "[3]")
        .unwrap();
    assert_eq!(fact.status, FactStatus::Ambiguous);
    assert_refuses_without_change(&source, "ambiguous");
}
