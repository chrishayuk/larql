//! A weight edit as a portable `.vlp` instruction: saved, applied in a fresh
//! session, removed, stacked, refused when it does not belong.

use larql_vindex::format::vindex3::weight_patch::bind_source;

use crate::support::*;

/// A change applied to a parsed `.vlp`, to build a corrupted one.
type Change = Box<dyn Fn(&mut serde_json::Value)>;

fn compile(s: &mut larql_lql::Session, out: &std::path::Path) {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)));
}

#[test]
fn a_weight_edit_is_saved_with_its_coordinates_and_source_identity() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let patch: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&vlp).unwrap()).unwrap();
    assert_eq!(
        patch["version"], 2,
        "a patch carrying stored-weight edits is version 2"
    );
    let ops = patch["operations"].as_array().unwrap();
    assert_eq!(ops.len(), 1);
    let op = &ops[0];
    assert_eq!(op["op"], "weight_delta");
    assert_eq!(op["tensor"], TENSOR);
    let saved: Vec<(u64, u64, u64)> = op["edits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["row"].as_u64().unwrap(),
                e["col"].as_u64().unwrap(),
                e["delta_bits"].as_u64().unwrap(),
            )
        })
        .collect();
    let bits = 0.0625f32.to_bits() as u64;
    assert_eq!(
        saved,
        COLS.iter()
            .map(|&c| (ROW as u64, c as u64, bits))
            .collect::<Vec<_>>()
    );
    // Bound to the exact artifact, not to the model name.
    let bound = bind_source(&fx.container, TENSOR).unwrap();
    assert_eq!(op["source"]["index_sha256"], bound.index_sha256);
    assert_eq!(op["source"]["representation"], CANONICAL);
    assert_eq!(
        op["source"]["payload_sha256"],
        digests(&fx.container)[CANONICAL].0
    );
    assert_eq!(op["digest"].as_str().unwrap().len(), 64);
}

#[test]
fn applying_in_a_fresh_session_reproduces_the_directly_compiled_bytes() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);

    let mut direct = session_on(&fx.container);
    run(&mut direct, &update(&COLS, DELTA));
    let direct_out = fx.path("direct.vindex3");
    compile(&mut direct, &direct_out);

    let mut fresh = session_on(&fx.container);
    let applied = run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    assert!(
        applied[0].contains("4 stored-weight edit(s)"),
        "{applied:?}"
    );
    let patch_out = fx.path("patched.vindex3");
    compile(&mut fresh, &patch_out);

    let (a, b) = (digests(&direct_out), digests(&patch_out));
    for id in [CANONICAL, DERIVED] {
        assert_eq!(
            a[id], b[id],
            "{id}: the same bytes whether edited directly or by patch"
        );
    }
    assert!(is_coherent(&patch_out));
    assert_eq!(
        ledger(&direct_out)["origins"],
        ledger(&patch_out)["origins"],
        "the same op is the origin"
    );
}

#[test]
fn removing_the_patch_restores_the_original_bytes() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    assert!(pending_line(&mut s)
        .unwrap()
        .contains("4 element(s) in 1 op(s)"));
    run(&mut s, &format!("REMOVE PATCH \"{}\";", q(&vlp)));
    assert!(
        pending_line(&mut s).is_none(),
        "no stored-weight edits remain"
    );

    let out = fx.path("reverted.vindex3");
    compile(&mut s, &out);
    let (src, now) = (digests(&fx.container), digests(&out));
    for id in [CANONICAL, DERIVED] {
        assert_eq!(src[id], now[id], "{id} is the original, bit for bit");
    }
    assert!(!out.join("weight_patches.json").exists());
}

#[test]
fn removing_a_patch_leaves_directly_staged_edits_alone() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[100], DELTA));
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    run(&mut s, &format!("REMOVE PATCH \"{}\";", q(&vlp)));
    assert!(pending_line(&mut s)
        .unwrap()
        .contains("1 element(s) in 1 op(s)"));
}

#[test]
fn applying_the_same_patch_twice_is_refused_and_never_doubles_the_edit() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    let err = try_run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp))).unwrap_err();
    assert!(err.contains("already edited"), "{err}");
    assert!(pending_line(&mut s)
        .unwrap()
        .contains("4 element(s) in 1 op(s)"));
    let out = fx.path("once.vindex3");
    compile(&mut s, &out);
    assert_eq!(
        ledger(&out)["targets"][0]["edits"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
}

#[test]
fn a_patch_conflicts_with_a_direct_edit_of_the_same_element() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[34], "0.5"));
    let err = try_run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp))).unwrap_err();
    assert!(
        err.contains("already edited") && err.contains("this session"),
        "{err}"
    );
    assert!(
        pending_line(&mut s)
            .unwrap()
            .contains("1 element(s) in 1 op(s)"),
        "the patch was not partly applied"
    );
}

#[test]
fn a_patch_written_for_another_revision_of_the_model_is_refused() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    // Same model identity, different bytes: an edited container is another revision.
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&[200], DELTA));
    let other = fx.path("other.vindex3");
    compile(&mut s, &other);

    let mut target = session_on(&other);
    let err = try_run(&mut target, &format!("APPLY PATCH \"{}\";", q(&vlp))).unwrap_err();
    assert!(err.contains("another artifact"), "{err}");
    assert!(pending_line(&mut target).is_none());
}

#[test]
fn a_corrupted_or_edited_patch_is_refused_and_applies_nothing() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let cases: Vec<(&str, Change)> = vec![
        (
            "a delta changed",
            Box::new(|v| v["operations"][0]["edits"][0]["delta_bits"] = 0x3e800000u32.into()),
        ),
        (
            "the digest changed",
            Box::new(|v| v["operations"][0]["digest"] = "0".repeat(64).into()),
        ),
        (
            "the tensor changed",
            Box::new(|v| v["operations"][0]["tensor"] = "0.mlp.down_proj.weight".into()),
        ),
        (
            "the source weights changed",
            Box::new(|v| v["operations"][0]["source"]["payload_sha256"] = "1".repeat(64).into()),
        ),
        (
            "the rounding rule changed",
            Box::new(|v| v["operations"][0]["rounding"] = "truncate".into()),
        ),
        (
            "an element removed",
            Box::new(|v| {
                v["operations"][0]["edits"].as_array_mut().unwrap().pop();
            }),
        ),
    ];
    for (what, change) in cases {
        let bad = mutated(&fx, &vlp, "bad.vlp", |v| change(v));
        let mut s = session_on(&fx.container);
        let err = try_run(&mut s, &format!("APPLY PATCH \"{}\";", q(&bad))).unwrap_err();
        assert!(err.contains("failed to apply patch"), "{what}: {err}");
        assert!(
            pending_line(&mut s).is_none(),
            "{what}: nothing may be applied"
        );
    }
}

#[test]
fn a_truncated_or_unknown_patch_does_not_load() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let text = std::fs::read_to_string(&vlp).unwrap();
    std::fs::write(fx.path("cut.vlp"), &text[..text.len() / 2]).unwrap();
    let unknown = mutated(&fx, &vlp, "unknown.vlp", |v| {
        v["operations"][0]["op"] = "weight_zap".into()
    });
    let mut s = session_on(&fx.container);
    for p in [fx.path("cut.vlp"), unknown] {
        let err = try_run(&mut s, &format!("APPLY PATCH \"{}\";", q(&p))).unwrap_err();
        assert!(err.contains("failed to load patch"), "{err}");
    }
    assert!(pending_line(&mut s).is_none());
}

#[test]
fn two_patches_touching_different_elements_stack() {
    let fx = Fixture::new();
    let (p1, p2) = (
        save_patch(&fx, "one.vlp", &COLS, DELTA),
        save_patch(&fx, "two.vlp", &[100, 101], "-0.125"),
    );
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&p1)));
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&p2)));
    assert!(pending_line(&mut s)
        .unwrap()
        .contains("6 element(s) in 2 op(s)"));
    let out = fx.path("stacked.vindex3");
    compile(&mut s, &out);
    let (a, b) = (
        tensor_bytes(&fx.container, CANONICAL, TENSOR),
        tensor_bytes(&out, CANONICAL, TENSOR),
    );
    assert_eq!(
        elements_differing(&a, &b),
        6,
        "both edits are retained, and nothing else changes"
    );
    assert_eq!(ledger(&out)["origins"].as_array().unwrap().len(), 2);
}

#[test]
fn a_conflicting_patch_is_refused_and_the_first_stays_applied() {
    let fx = Fixture::new();
    let (p1, p3) = (
        save_patch(&fx, "one.vlp", &COLS, DELTA),
        save_patch(&fx, "three.vlp", &[33, 200], DELTA),
    );
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&p1)));
    let err = try_run(&mut s, &format!("APPLY PATCH \"{}\";", q(&p3))).unwrap_err();
    assert!(err.contains("already edited"), "{err}");
    assert!(pending_line(&mut s)
        .unwrap()
        .contains("4 element(s) in 1 op(s)"));
}

#[test]
fn restarting_and_reapplying_gives_an_identical_artifact() {
    let fx = Fixture::new();
    let vlp = save_patch(&fx, "a.vlp", &COLS, DELTA);
    let mut outs = Vec::new();
    for n in 0..2 {
        let mut s = session_on(&fx.container);
        run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
        let out = fx.path(&format!("run{n}.vindex3"));
        compile(&mut s, &out);
        outs.push(out);
    }
    assert_eq!(digests(&outs[0]), digests(&outs[1]));
    assert_eq!(ledger(&outs[0]), ledger(&outs[1]));
}

#[test]
fn recording_a_patch_session_with_no_edits_still_saves_a_version_1_patch() {
    let fx = Fixture::new();
    let mut s = session_on(&fx.container);
    let vlp = fx.path("empty.vlp");
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(&mut s, "SAVE PATCH;");
    let patch: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&vlp).unwrap()).unwrap();
    assert_eq!(
        patch["version"], 1,
        "patches without weight edits keep the existing version"
    );
}
