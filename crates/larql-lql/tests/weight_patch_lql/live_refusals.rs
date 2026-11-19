//! Refusals: while a stored-weight op is pending and cannot be executed exactly, `INFER`, `INFER … GENERATE`
//! and `TRACE` refuse and say why. They never run the stored weights in its place.
//!
//! Two rules are pinned in `larql-vindex`'s `weight_patch` tests and not here, because LQL cannot stage the op
//! they concern: a tensor of an expert bank and a tensor the plan does not project through are refused when
//! the op is written (`UPDATE WEIGHTS` needs a BF16 matrix of the bound container's decoder), so nothing is
//! ever pending.

use larql_lql::{RepresentationSource, Session};

use crate::live_support::*;
use crate::support::*;

/// A directory a container does not hold, which COMPILE refuses to carry.
const STRAY_DIRECTORY: &str = "notes";

/// Every execution statement, and that each refuses with `reason`.
fn assert_execution_refused(s: &mut Session, reason: &str) {
    for statement in [
        infer_statement(),
        format!("INFER \"{PROMPT}\" GENERATE {NEW_TOKENS};"),
        format!("TRACE \"{PROMPT}\";"),
    ] {
        let error = try_run(s, &statement).unwrap_err();
        assert!(
            error.contains("stored-weight ops are pending in this session and cannot be executed")
                && error.contains("the stored weights will not be executed in their place")
                && error.contains(reason),
            "{statement}: {error}"
        );
    }
}

#[test]
fn a_session_whose_source_may_manufacture_a_representation_refuses_to_execute() {
    for (source, name) in [
        (RepresentationSource::Auto, "auto"),
        (RepresentationSource::Transient, "transient"),
    ] {
        let fx = inferable();
        let mut s = Session::new();
        s.set_representation_source(source);
        run(&mut s, &format!("USE \"{}\";", q(&fx.container)));
        let base = infer(&mut s);

        let staged = run(
            &mut s,
            &statement(&fx, Kind::UpdateWeights, COLUMNS_A, "unused.json"),
        );
        assert!(
            has(&staged, "NOT executable in this session")
                && has(&staged, &format!("representation source is `{name}`")),
            "{staged:?}"
        );
        assert!(live_digests(&staged).is_empty());
        assert_execution_refused(&mut s, "execute only under `stored`");

        // The op is still pending and still compiles: only executing it here is refused.
        let out = fx.path("compiled.vindex3");
        run(
            &mut s,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
        );
        let mut fresh = Session::new();
        fresh.set_representation_source(source);
        run(&mut fresh, &format!("USE \"{}\";", q(&out)));
        assert_ne!(infer(&mut fresh), base);
    }
}

#[test]
fn knowledge_edits_and_weight_ops_together_refuse_to_execute() {
    let fx = inferable();
    let mut s = session_on(&fx.container);
    run(
        &mut s,
        &statement(&fx, Kind::UpdateWeights, COLUMNS_A, "unused.json"),
    );
    infer(&mut s);
    // A KNN install captures its key over the program the session executes: the patched one.
    let inserted = run(
        &mut s,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]");"#,
    );
    assert!(has(&inserted, "Inserted"), "{inserted:?}");
    assert_execution_refused(&mut s, "COMPILE refuses that mix");
    let shown = run(&mut s, "SHOW PATCHES;");
    assert!(has(&shown, "NOT executable in this session"), "{shown:?}");

    // Knowledge first, then the weight op: the staging reply already says so.
    let mut s = session_on(&fx.container);
    run(&mut s, "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;");
    let staged = run(
        &mut s,
        &statement(&fx, Kind::UpdateWeights, COLUMNS_B, "unused.json"),
    );
    assert!(has(&staged, "COMPILE refuses that mix"), "{staged:?}");
    assert_execution_refused(&mut s, "COMPILE refuses that mix");
}

#[test]
fn a_compose_install_is_refused_while_a_weight_op_is_pending() {
    let fx = inferable();
    let mut s = session_on(&fx.container);
    run(
        &mut s,
        &statement(&fx, Kind::UpdateWeights, COLUMNS_A, "unused.json"),
    );
    let patched = infer(&mut s);
    let error = try_run(
        &mut s,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "rel", "[2]") MODE COMPOSE;"#,
    )
    .unwrap_err();
    assert!(error.contains("INSERT MODE COMPOSE refused"), "{error}");
    assert_eq!(
        infer(&mut s),
        patched,
        "the refused install left the session as it was"
    );
}

/// A pre-write refusal of COMPILE that staging does not make: the container holds a directory a patch would
/// not know how to carry. The op stages, COMPILE refuses, and so does execution, with COMPILE's reason.
#[test]
fn ops_compile_would_refuse_refuse_to_execute_with_its_reason() {
    let fx = inferable();
    std::fs::create_dir(fx.container.join(STRAY_DIRECTORY)).unwrap();
    let mut s = session_on(&fx.container);
    let staged = run(
        &mut s,
        &statement(&fx, Kind::UpdateWeights, COLUMNS_A, "unused.json"),
    );
    assert!(
        has(&staged, "NOT executable in this session")
            && has(&staged, "COMPILE would refuse these ops"),
        "{staged:?}"
    );
    assert_execution_refused(&mut s, STRAY_DIRECTORY);
    let compile = try_run(
        &mut s,
        &format!(
            "COMPILE CURRENT INTO VINDEX \"{}\";",
            q(fx.path("refused.vindex3"))
        ),
    )
    .unwrap_err();
    assert!(compile.contains(STRAY_DIRECTORY), "{compile}");
}

/// A patch that carried the unexecutable op is removed: execution is the base's again.
#[test]
fn removing_the_patch_lifts_the_refusal() {
    let fx = inferable();
    let vlp = save_vlp(&fx, Kind::UpdateWeights, COLUMNS_A, "a");
    // The tombstone comes from an applied patch: REMOVE PATCH refuses while
    // a session edit no applied patch holds is live (Gate B, B2).
    let tombstone = fx.path("tombstone.vlp");
    let mut author = session_on(&fx.container);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", q(&tombstone)));
    run(
        &mut author,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    run(&mut author, "SAVE PATCH;");
    let mut s = session_on(&fx.container);
    apply(&mut s, &tombstone);
    let base = printed(&mut s);
    let applied = apply(&mut s, &vlp);
    assert!(
        has(&applied, "NOT executable in this session"),
        "{applied:?}"
    );
    assert_execution_refused(&mut s, "COMPILE refuses that mix");
    let removed = remove(&mut s, &vlp);
    assert!(!has(&removed, "NOT executable"), "{removed:?}");
    // The tombstone's patch stays applied; only the weight op went.
    assert_eq!(printed(&mut s), base);
}
