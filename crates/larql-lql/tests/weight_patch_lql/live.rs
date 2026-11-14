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
            has(&applied, "In effect for INFER and TRACE in this session"),
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
            has(&staged, "In effect for INFER and TRACE in this session"),
            "{kind:?}: {staged:?}"
        );
        assert_ne!(infer(&mut s), base, "{kind:?}");
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
        "In effect for INFER and TRACE in this session: 1 tensor(s) execute from patched bytes held in memory, \
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
        has(&shown, "pending COMPILE") && has(&shown, "In effect for INFER and TRACE"),
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
