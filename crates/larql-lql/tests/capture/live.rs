//! A capture runs whatever the session executes: under an applied patch, the patched weights, equal to a capture
//! of the container COMPILE writes from the patch; after REMOVE PATCH, the stored weights again.

use crate::support::*;

fn logits(s: &mut larql_lql::Session, fx: &Fixture, name: &str) -> (Vec<u32>, serde_json::Value) {
    let out = fx.path(name);
    capture(s, &format!("logits FOR \"{PROMPT}\" POSITIONS ALL"), &out);
    (bits(&read_npy(&out).values), sidecar(&out))
}

#[test]
fn a_patch_is_captured_from_apply_until_remove_and_equals_the_compiled_container() {
    let fx = Fixture::new();
    let vlp = fx.patch("edit.vlp");
    let mut s = fx.session();
    let (base, base_car) = logits(&mut s, &fx, "base.npy");
    assert_eq!(base_car["executes"], "stored");
    assert_eq!(base_car["patches"], serde_json::json!([]));

    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    let (patched, car) = logits(&mut s, &fx, "patched.npy");
    assert_ne!(patched, base, "the patch changes what is captured");
    assert_eq!(car["executes"], "patched");
    let patch = &car["patches"][0];
    // The sidecar records the file path itself; `q` is only the LQL-literal escaping used to name it.
    assert_eq!(patch["path"], vlp.display().to_string());
    assert_eq!(patch["sha256"], sha256_hex(&std::fs::read(&vlp).unwrap()));
    let op = &patch["weight_ops"][0];
    assert_eq!(op["kind"], "weight_delta");
    assert!(op["tensor"].as_str().unwrap().ends_with(TENSOR), "{op}");
    let recorded: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&vlp).unwrap()).unwrap();
    let digest = recorded.to_string();
    assert!(
        digest.contains(op["digest"].as_str().unwrap()),
        "the op digest is the .vlp's own"
    );
    assert_eq!(car["live_tensors"][0]["tensor"], TENSOR);

    let compiled = fx.path("compiled.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&compiled)),
    );
    let mut fresh = session_on(&compiled);
    let (from_compiled, compiled_car) = logits(&mut fresh, &fx, "compiled.npy");
    assert_eq!(
        from_compiled, patched,
        "the live capture is the compiled container's"
    );
    assert_eq!(compiled_car["executes"], "stored");
    assert_eq!(
        compiled_car["live_tensors"],
        serde_json::json!([]),
        "the compiled container stores those bytes"
    );

    run(&mut s, &format!("REMOVE PATCH \"{}\";", q(&vlp)));
    let (after, after_car) = logits(&mut s, &fx, "after.npy");
    assert_eq!(after, base, "after the removal, the base exactly");
    assert_eq!(after_car["executes"], "stored");
    assert_eq!(after_car["patches"], serde_json::json!([]));
}

#[test]
fn a_staged_op_is_captured_and_named_as_the_sessions() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let (base, _) = logits(&mut s, &fx, "base.npy");
    run(&mut s, STAGE);
    let (staged, car) = logits(&mut s, &fx, "staged.npy");
    assert_ne!(staged, base);
    assert_eq!(car["executes"], "patched");
    assert_eq!(car["patches"][0]["source"], "this session");
    assert_eq!(car["patches"][0]["weight_ops"][0]["kind"], "weight_delta");
    run(&mut s, "DISCARD WEIGHT EDITS;");
    assert_eq!(logits(&mut s, &fx, "discarded.npy").0, base);
}

#[test]
fn the_down_projection_is_captured_under_the_patch_too() {
    let fx = Fixture::new();
    let vlp = fx.patch("edit.vlp");
    let site = format!("ffn.down_output AT LAYER 1 FOR \"{PROMPT}\" POSITIONS ALL");
    let mut s = fx.session();
    let base = fx.path("base.npy");
    capture(&mut s, &site, &base);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    let patched = fx.path("patched.npy");
    capture(&mut s, &site, &patched);
    assert_ne!(
        bits(&read_npy(&base).values),
        bits(&read_npy(&patched).values)
    );
    assert_eq!(
        sidecar(&patched)["execution"]["cross_check"]["bit_identical"],
        true
    );

    let compiled = fx.path("compiled.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&compiled)),
    );
    let mut fresh = session_on(&compiled);
    let from_compiled = fx.path("compiled.npy");
    capture(&mut fresh, &site, &from_compiled);
    assert_eq!(
        bits(&read_npy(&from_compiled).values),
        bits(&read_npy(&patched).values)
    );
}
