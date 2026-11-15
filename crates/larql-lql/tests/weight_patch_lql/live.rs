//! Session-scoped execution of stored-weight ops: from `APPLY PATCH` (or a staging statement) until
//! `REMOVE PATCH`, `INFER`, `INFER … GENERATE` and `TRACE` compute with the patched weights, and with the
//! stored weights again afterwards. Nothing on disk changes, and what the session computes meanwhile is what
//! a fresh binding of the compiled container computes.

use crate::live_support::*;
use crate::support::*;

/// How many times the stability test repeats one `INFER`.
const REPEATS: usize = 3;

fn compile(s: &mut larql_lql::Session, out: &std::path::Path) {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)));
}

#[test]
fn a_patch_executes_from_apply_until_remove_and_the_base_returns_exactly() {
    for kind in KINDS {
        let fx = inferable();
        let vlp = save_vlp(&fx, kind, COLUMNS_A, "a");
        let on_disk = fingerprint(&fx.container);
        let mut s = session_on(&fx.container);
        let base = printed(&mut s);

        let applied = apply(&mut s, &vlp);
        assert!(
            has(
                &applied,
                "In effect for INFER, TRACE and CAPTURE in this session"
            ),
            "{kind:?}: {applied:?}"
        );
        let patched = printed(&mut s);
        assert_ne!(
            patched.infer, base.infer,
            "{kind:?}: the patch changes what INFER prints"
        );

        let removed = remove(&mut s, &vlp);
        assert!(
            has(&removed, "No stored-weight op is in effect"),
            "{kind:?}: {removed:?}"
        );
        assert_eq!(
            printed(&mut s),
            base,
            "{kind:?}: after removal, the base exactly"
        );
        assert_eq!(
            fingerprint(&fx.container),
            on_disk,
            "{kind:?}: nothing on disk changed"
        );
    }
}

#[test]
fn a_directly_staged_op_executes_too() {
    for kind in KINDS {
        let fx = inferable();
        let mut s = session_on(&fx.container);
        let base = infer(&mut s);
        let staged = run(&mut s, &statement(&fx, kind, COLUMNS_A, "direct.json"));
        assert!(
            has(
                &staged,
                "In effect for INFER, TRACE and CAPTURE in this session"
            ),
            "{kind:?}: {staged:?}"
        );
        assert_ne!(infer(&mut s), base, "{kind:?}");
    }
}

#[test]
fn discarding_staged_ops_stops_them_executing() {
    for kind in KINDS {
        let fx = inferable();
        let mut s = session_on(&fx.container);
        let base = printed(&mut s);
        run(&mut s, &statement(&fx, kind, COLUMNS_A, "direct.json"));
        assert_ne!(
            printed(&mut s).infer,
            base.infer,
            "{kind:?}: the staged op executes"
        );

        let discarded = run(&mut s, "DISCARD WEIGHT EDITS;");
        assert!(
            has(&discarded, "No stored-weight op is in effect"),
            "{kind:?}: {discarded:?}"
        );
        assert_eq!(
            printed(&mut s),
            base,
            "{kind:?}: after the discard, the base exactly"
        );
    }
}

#[test]
fn a_discard_leaves_an_applied_patch_executing_and_nothing_of_the_discarded_op() {
    // The case a view left standing would get wrong: one op from a patch, one staged, the staged one discarded.
    for kind in KINDS {
        let fx = inferable();
        let b = save_vlp(&fx, kind, COLUMNS_B, "b");
        let mut only_b = session_on(&fx.container);
        apply(&mut only_b, &b);
        let want = printed(&mut only_b);

        let mut s = session_on(&fx.container);
        apply(&mut s, &b);
        run(&mut s, &statement(&fx, kind, COLUMNS_A, "direct.json"));
        let both = printed(&mut s);
        assert_ne!(
            both.infer, want.infer,
            "{kind:?}: the staged op adds to the patch"
        );

        let discarded = run(&mut s, "DISCARD WEIGHT EDITS;");
        assert!(
            has(
                &discarded,
                "In effect for INFER, TRACE and CAPTURE in this session"
            ),
            "{kind:?}: the patch's op is still in effect: {discarded:?}"
        );
        assert_eq!(
            printed(&mut s),
            want,
            "{kind:?}: what executes is the patch alone"
        );
    }
}

/// One session through a discard and everything after it, each state held to a reference built independently
/// in a session of its own: B alone, B with A staged, and the base.
///
/// ```text
/// USE; APPLY PATCH b; stage A; DISCARD WEIGHT EDITS   -> B alone, and not B with A
/// REMOVE PATCH b                                      -> the base
/// APPLY PATCH b                                       -> B alone again
/// stage A                                             -> B with A again
/// ```
#[test]
fn a_session_through_a_discard_matches_independent_references_at_every_step() {
    for kind in KINDS {
        let fx = inferable();
        let b = save_vlp(&fx, kind, COLUMNS_B, "b");
        let on_disk = fingerprint(&fx.container);
        let stage_a = statement(&fx, kind, COLUMNS_A, "direct.json");

        // The references, each from a session that did only what its name says.
        let mut untouched = session_on(&fx.container);
        let base = printed(&mut untouched);
        let mut reference = session_on(&fx.container);
        apply(&mut reference, &b);
        let only_b = printed(&mut reference);
        let mut reference = session_on(&fx.container);
        apply(&mut reference, &b);
        run(&mut reference, &stage_a);
        let both = printed(&mut reference);
        // Three different programs: every equality below is then a claim about which one ran.
        assert_ne!(only_b.infer, base.infer, "{kind:?}: B changes the base");
        assert_ne!(both.infer, only_b.infer, "{kind:?}: A adds to B");
        assert_ne!(both.infer, base.infer, "{kind:?}");

        let mut s = session_on(&fx.container);
        apply(&mut s, &b);
        run(&mut s, &stage_a);
        assert_eq!(
            printed(&mut s),
            both,
            "{kind:?}: before the discard, both ops execute"
        );

        // 1. The discard: B alone. Equal to `only_b` fails if A still executes, and if the stored weights do.
        let discarded = run(&mut s, "DISCARD WEIGHT EDITS;");
        assert!(
            has(&discarded, "Discarded: 1 stored-weight op(s)"),
            "{kind:?}: {discarded:?}"
        );
        let after_discard = printed(&mut s);
        assert_eq!(
            after_discard, only_b,
            "{kind:?}: after the discard, B alone"
        );
        assert_ne!(
            after_discard.infer, both.infer,
            "{kind:?}: nothing of A executes"
        );
        assert_ne!(
            after_discard.infer, base.infer,
            "{kind:?}: and B still does"
        );

        // 2. Removing B: the base exactly.
        remove(&mut s, &b);
        assert_eq!(
            printed(&mut s),
            base,
            "{kind:?}: after the removal, the base"
        );

        // 3. B again: nothing left behind by the discard or the removal, and nothing lost.
        apply(&mut s, &b);
        assert_eq!(
            printed(&mut s),
            only_b,
            "{kind:?}: B applied again is B alone"
        );

        // 4. A again: its elements and columns were freed, and it executes beside B as it did.
        run(&mut s, &stage_a);
        assert_eq!(
            printed(&mut s),
            both,
            "{kind:?}: A staged again executes beside B"
        );

        // 5. A session that applied nothing saw none of it, and neither did the disk.
        assert_eq!(
            printed(&mut untouched),
            base,
            "{kind:?}: the other session is untouched"
        );
        assert_eq!(
            fingerprint(&fx.container),
            on_disk,
            "{kind:?}: nothing on disk changed"
        );
    }
}

/// A second `USE` binds afresh: the ops staged and the patches applied before it are gone with the old
/// binding, and what executes is the stored weights of whatever is bound now.
#[test]
fn a_second_use_drops_every_pending_op_and_the_stored_weights_execute() {
    for kind in KINDS {
        let fx = inferable();
        let b = save_vlp(&fx, kind, COLUMNS_B, "b");
        let rebind = format!("USE \"{}\";", q(&fx.container));
        let mut s = session_on(&fx.container);
        let base = printed(&mut s);
        apply(&mut s, &b);
        run(&mut s, &statement(&fx, kind, COLUMNS_A, "direct.json"));
        assert_ne!(
            printed(&mut s).infer,
            base.infer,
            "{kind:?}: both ops execute"
        );

        run(&mut s, &rebind);
        assert!(
            pending_line(&mut s).is_none(),
            "{kind:?}: nothing is pending"
        );
        assert_eq!(
            printed(&mut s),
            base,
            "{kind:?}: after the second USE, the base"
        );
        // And the new binding takes the same ops again, as a fresh session does.
        apply(&mut s, &b);
        assert_ne!(printed(&mut s).infer, base.infer, "{kind:?}");
    }
}

#[test]
fn switching_patches_leaves_nothing_of_the_first_behind() {
    for kind in KINDS {
        let fx = inferable();
        let (a, b) = (
            save_vlp(&fx, kind, COLUMNS_A, "a"),
            save_vlp(&fx, kind, COLUMNS_B, "b"),
        );
        let mut only_b = session_on(&fx.container);
        apply(&mut only_b, &b);
        let want = printed(&mut only_b);

        let mut s = session_on(&fx.container);
        apply(&mut s, &a);
        let with_a = printed(&mut s);
        remove(&mut s, &a);
        apply(&mut s, &b);
        let with_b = printed(&mut s);
        assert_ne!(
            with_a.infer, with_b.infer,
            "{kind:?}: A and B are different edits"
        );
        assert_eq!(with_b, want, "{kind:?}: A then B is B alone");
    }
}

#[test]
fn a_patched_session_prints_the_same_thing_every_time() {
    for kind in KINDS {
        let fx = inferable();
        let vlp = save_vlp(&fx, kind, COLUMNS_A, "a");
        let mut s = session_on(&fx.container);
        apply(&mut s, &vlp);
        let first = infer(&mut s);
        for _ in 1..REPEATS {
            assert_eq!(infer(&mut s), first, "{kind:?}");
        }
    }
}

#[test]
fn two_sessions_on_one_container_do_not_see_each_others_patches() {
    for kind in KINDS {
        let fx = inferable();
        let vlp = save_vlp(&fx, kind, COLUMNS_A, "a");
        let mut patched = session_on(&fx.container);
        let mut clean = session_on(&fx.container);
        let base = infer(&mut clean);
        apply(&mut patched, &vlp);
        assert_ne!(infer(&mut patched), base, "{kind:?}");
        assert_eq!(
            infer(&mut clean),
            base,
            "{kind:?}: the other session is untouched"
        );
    }
}

/// Parity with compile. The bytes: each live tensor's digest, as the session reports it, is the digest of
/// that tensor's bytes in the container the same session compiles. The output: the session prints what a fresh
/// binding of that container prints.
#[test]
fn a_patched_session_matches_a_fresh_binding_of_the_container_it_compiles() {
    for kind in KINDS {
        let fx = inferable();
        let vlp = save_vlp(&fx, kind, COLUMNS_A, "a");
        let mut s = session_on(&fx.container);
        let base = infer(&mut s);
        let applied = apply(&mut s, &vlp);
        let live = printed(&mut s);

        let out = fx.path("compiled.vindex3");
        compile(&mut s, &out);
        assert_eq!(
            printed(&mut s),
            live,
            "{kind:?}: compiling changes nothing the session computes"
        );

        let stored = tensor_bytes(&out, CANONICAL, TENSOR);
        assert_ne!(
            stored,
            tensor_bytes(&fx.container, CANONICAL, TENSOR),
            "{kind:?}"
        );
        for reported in [&applied, &run(&mut s, "SHOW PATCHES;")] {
            assert_eq!(
                live_digests(reported),
                vec![sha256(&stored)],
                "{kind:?}: {reported:?}"
            );
        }

        let mut fresh = session_on(&out);
        assert!(live_digests(&run(&mut fresh, "SHOW PATCHES;")).is_empty());
        assert_eq!(printed(&mut fresh), live, "{kind:?}");
        assert_ne!(live.infer, base, "{kind:?}: and neither is the base");
    }
}

/// The patched bytes are produced when the patch is applied, not by each `INFER`. After the apply, the
/// container is given a directory COMPILE refuses to carry, so materialising again would be refused: `INFER`
/// still runs, on the bytes it already holds. The next change to the op set rebuilds the view and meets the
/// refusal.
#[test]
fn infer_does_not_materialise_again() {
    const STRAY_DIRECTORY: &str = "notes";
    let fx = inferable();
    let (a, b) = (
        save_vlp(&fx, Kind::UpdateWeights, COLUMNS_A, "a"),
        save_vlp(&fx, Kind::UpdateWeights, COLUMNS_B, "b"),
    );
    let mut s = session_on(&fx.container);
    apply(&mut s, &a);
    let patched = infer(&mut s);

    std::fs::create_dir(fx.container.join(STRAY_DIRECTORY)).unwrap();
    assert_eq!(infer(&mut s), patched);
    let applied = apply(&mut s, &b);
    assert!(
        has(&applied, "NOT executable in this session") && has(&applied, STRAY_DIRECTORY),
        "{applied:?}"
    );
    assert!(try_run(&mut s, &infer_statement()).is_err());
}

/// The sham control. A declared sham executes the base's bytes; an op that is not zero-valued and changes
/// nothing is refused when it is staged, so nothing is pending and the base executes.
#[test]
fn a_declared_sham_executes_the_base_and_an_op_that_changes_nothing_stays_refused() {
    const F64_ONE_HEX: &str = "3ff0000000000000";
    const F64_ZERO_HEX: &str = "0000000000000000";
    /// 2^-60: far below half a BF16 step of any stored element.
    const F64_TINY_HEX: &str = "3c30000000000000";
    const COLUMN: usize = 33;
    let fx = inferable();
    let file = |name: &str, a: &str| {
        let path = fx.path(name);
        let doc = serde_json::json!({
            "rank": 1, "a": vec![a; larql_vindex::format::vindex3::fixtures::DENSE_HIDDEN], "b": [F64_ONE_HEX],
        });
        std::fs::write(&path, doc.to_string()).unwrap();
        format!(
            "ADD LOW RANK \"{}\" TO \"{TENSOR}\" COLUMNS ({COLUMN}) NUMERICS \"{PROFILE}\";",
            q(path)
        )
    };
    let mut s = session_on(&fx.container);
    let base = printed(&mut s);

    let refused = try_run(&mut s, &file("tiny.json", F64_TINY_HEX)).unwrap_err();
    assert!(refused.contains("ADD LOW RANK refused"), "{refused}");
    assert!(
        pending_line(&mut s).is_none(),
        "a refused op is not pending"
    );
    assert_eq!(printed(&mut s), base);

    let staged = run(&mut s, &file("zero.json", F64_ZERO_HEX));
    assert!(has(&staged, "declared sham"), "{staged:?}");
    assert_eq!(
        live_digests(&staged),
        vec![sha256(&tensor_bytes(&fx.container, CANONICAL, TENSOR))],
        "the sham's live bytes are the base's"
    );
    assert_eq!(printed(&mut s), base);
}

/// What the replies say: in effect for INFER, nothing on disk, and no longer the stored bytes.
#[test]
fn the_replies_say_what_executes_and_that_nothing_on_disk_changed() {
    let fx = inferable();
    let mut s = session_on(&fx.container);
    let staged = run(
        &mut s,
        &statement(&fx, Kind::UpdateWeights, COLUMNS_A, "unused.json"),
    );
    for needle in [
        "In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, \
         not from the container's stored bytes.",
        "Nothing on disk has changed: COMPILE CURRENT INTO VINDEX writes a container that stores exactly these bytes.",
        "DESCRIBE, WALK, SELECT and DIFF read the stored weights.",
        TENSOR,
    ] {
        assert!(has(&staged, needle), "missing {needle:?} in {staged:?}");
    }
    assert!(!has(&staged, "does not see it"), "{staged:?}");
    let shown = run(&mut s, "SHOW PATCHES;");
    assert!(
        has(&shown, "pending COMPILE") && has(&shown, "In effect for INFER, TRACE and CAPTURE"),
        "{shown:?}"
    );

    // A knowledge-only patch says nothing about weights when it is applied or removed.
    let knowledge = fx.path("knowledge.vlp");
    let mut author = session_on(&fx.container);
    run(&mut author, &format!("BEGIN PATCH \"{}\";", q(&knowledge)));
    run(
        &mut author,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    run(&mut author, "SAVE PATCH;");
    let mut clean = session_on(&fx.container);
    assert_eq!(apply(&mut clean, &knowledge).len(), 1);
    assert_eq!(remove(&mut clean, &knowledge).len(), 1);
}
